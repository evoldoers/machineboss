/**
 * 1D Forward dispatch using transfer-matrix parallel prefix scan on GPU.
 *
 * Steps:
 * 1. Compute silent closure on CPU (small, machine-dependent)
 * 2. Upload closure, transition slice, and tokens to GPU
 * 3. Build per-position transfer matrices (GPU kernel)
 * 4. Hillis-Steele prefix scan of S×S matrices (GPU kernel, O(log L) passes)
 * 5. Extract forward vectors from prefix products (GPU kernel)
 * 6. Read back result
 */

import {
  loadShader, createPipeline, createStorageBuffer,
  createEmptyBuffer, createUniformBuffer, createBindGroup,
  readBuffer,
} from './pipeline.mjs';
import { silentClosure } from '../cpu/silent.mjs';
import { makeSemiring, logIdentity, NEG_INF } from '../internal/logmath.mjs';

/**
 * 1D Forward on GPU using prefix scan.
 *
 * @param {GPUDevice} device
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq
 * @param {Uint32Array|null} outputSeq
 * @param {string} [semiringType='logsumexp']
 * @returns {Promise<number>} log-likelihood
 */
export async function forward1DGPU(device, machine, inputSeq, outputSeq, semiringType = 'logsumexp') {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans, logTransF32 } = machine;
  const semiring = makeSemiring(semiringType);
  const useMax = semiringType === 'maxplus' ? 1 : 0;

  let seq, isInput, nTok;
  if (inputSeq === null || inputSeq === undefined) {
    seq = outputSeq;
    isInput = false;
    nTok = nOut;
  } else {
    seq = inputSeq;
    isInput = true;
    nTok = nIn;
  }

  const L = seq.length;
  if (L === 0) {
    // Compute on CPU for empty sequences
    const silent = new Float64Array(S * S);
    for (let i = 0; i < S * S; i++) silent[i] = logTrans[i];
    const closure = silentClosure(silent, S, semiring.reduce);
    return closure[S - 1]; // closure[0][S-1]
  }

  // Step 1: Compute silent closure on CPU
  const silent64 = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) silent64[i] = logTrans[i];
  const closure64 = silentClosure(silent64, S, semiring.reduce);
  const closure32 = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++) closure32[i] = closure64[i];

  // Extract transition slice for active dimension (f32)
  const transSlice32 = new Float32Array(nTok * S * S);
  for (let tok = 0; tok < nTok; tok++) {
    for (let src = 0; src < S; src++) {
      for (let dst = 0; dst < S; dst++) {
        let idx;
        if (isInput) {
          idx = ((tok * nOut + 0) * S + src) * S + dst;
        } else {
          idx = ((0 * nOut + tok) * S + src) * S + dst;
        }
        transSlice32[(tok * S + src) * S + dst] = logTransF32[idx];
      }
    }
  }

  // Step 2: Upload to GPU
  const tokensBuffer = createStorageBuffer(device, seq);
  const transSliceBuffer = createStorageBuffer(device, transSlice32);
  const closureBuffer = createStorageBuffer(device, closure32);
  const transfersBuffer = createEmptyBuffer(device, L * S * S * 4);

  // Step 3: Build transfer matrices
  const buildShader = loadShader('transfer-build.wgsl');
  const buildPipeline = createPipeline(device, buildShader, { S, N_TOKENS: nTok });
  const buildBindGroup = createBindGroup(device, buildPipeline, 0,
    [tokensBuffer, transSliceBuffer, closureBuffer, transfersBuffer]);

  const encoder1 = device.createCommandEncoder();
  const pass1 = encoder1.beginComputePass();
  pass1.setPipeline(buildPipeline);
  pass1.setBindGroup(0, buildBindGroup);
  pass1.dispatchWorkgroups(L);
  pass1.end();
  device.queue.submit([encoder1.finish()]);

  // Step 4: Hillis-Steele prefix scan
  const scanShader = loadShader('prefix-scan.wgsl');

  // Double buffer
  let bufA = transfersBuffer;
  let bufB = createEmptyBuffer(device, L * S * S * 4);

  const nPasses = Math.ceil(Math.log2(L));
  let stride = 1;

  for (let pass = 0; pass < nPasses; pass++) {
    const paramsData = new Uint32Array([L, stride]);
    // Need to pad to 8 bytes minimum for uniform buffer
    const paramsBuffer = createUniformBuffer(device, new Uint32Array([L, stride]));

    const scanPipeline = createPipeline(device, scanShader, { S, USE_MAX: useMax, IS_BACKWARD: 0 });
    const scanBindGroup = createBindGroup(device, scanPipeline, 0, [bufA, bufB, paramsBuffer]);

    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(scanPipeline);
    p.setBindGroup(0, scanBindGroup);
    p.dispatchWorkgroups(L);
    p.end();
    device.queue.submit([enc.finish()]);

    // Swap buffers
    [bufA, bufB] = [bufB, bufA];
    stride *= 2;
    paramsBuffer.destroy();
  }

  // Result is in bufA (the last "output" buffer after swap)
  // Step 5: Extract forward vector at position L-1
  // fwd[dst] = logsumexp_src(init_closed[src] + prefix[L-1, src, dst])

  // Compute init_closed on CPU: closure applied to start vector
  const init = new Float32Array(S).fill(-Infinity);
  init[0] = 0.0;
  // init_closed[dst] = logsumexp_src(init[src] + closure[src * S + dst])
  const initClosed32 = new Float32Array(S);
  for (let dst = 0; dst < S; dst++) {
    let acc = -Infinity;
    for (let src = 0; src < S; src++) {
      const val = init[src] + closure32[src * S + dst];
      if (val > acc) {
        if (acc === -Infinity) {
          acc = val;
        } else {
          const m = val > acc ? val : acc;
          acc = m + Math.log(Math.exp(val - m) + Math.exp(acc - m));
        }
      }
    }
    initClosed32[dst] = acc;
  }

  // Read prefix product at position L-1
  const prefixData = await readBuffer(device, bufA, L * S * S * 4);

  // Extract on CPU (simpler than another GPU dispatch for small S)
  const prefixBase = (L - 1) * S * S;
  let logLikelihood = -Infinity;
  for (let dst = 0; dst < S; dst++) {
    let acc = -Infinity;
    for (let src = 0; src < S; src++) {
      const val = initClosed32[src] + prefixData[prefixBase + src * S + dst];
      if (val !== -Infinity) {
        if (acc === -Infinity) {
          acc = val;
        } else {
          const m = val > acc ? val : acc;
          acc = m + Math.log(Math.exp(val - m) + Math.exp(acc - m));
        }
      }
    }
    if (dst === S - 1) logLikelihood = acc;
  }

  // Cleanup
  tokensBuffer.destroy();
  transSliceBuffer.destroy();
  closureBuffer.destroy();
  transfersBuffer.destroy();
  if (bufA !== transfersBuffer) bufA.destroy();
  if (bufB !== transfersBuffer) bufB.destroy();

  return logLikelihood;
}
