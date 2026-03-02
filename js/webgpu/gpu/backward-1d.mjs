/**
 * 1D Backward dispatch using transfer-matrix parallel suffix scan on GPU.
 *
 * Same as forward but with IS_BACKWARD=1 in the prefix scan shader,
 * which reverses the matmul argument order.
 */

import {
  loadShader, createPipeline, createStorageBuffer,
  createEmptyBuffer, createUniformBuffer, createBindGroup,
  readBuffer,
} from './pipeline.mjs';
import { silentClosure } from '../cpu/silent.mjs';
import { makeSemiring, NEG_INF } from '../internal/logmath.mjs';

/**
 * 1D Backward on GPU using suffix scan.
 *
 * @param {GPUDevice} device
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq
 * @param {Uint32Array|null} outputSeq
 * @param {string} [semiringType='logsumexp']
 * @returns {Promise<{logLikelihood: number, bp: Float32Array}>}
 *   bp is (L+1, S) backward probabilities
 */
export async function backward1DGPU(device, machine, inputSeq, outputSeq, semiringType = 'logsumexp') {
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

  // Compute silent closure on CPU
  const silent64 = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) silent64[i] = logTrans[i];
  const closure64 = silentClosure(silent64, S, semiring.reduce);
  const closure32 = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++) closure32[i] = closure64[i];

  // Terminal vector: end state = S-1
  // term_closed[src] = logsumexp_dst(closure[src * S + dst] + term[dst])
  const termClosed32 = new Float32Array(S);
  for (let src = 0; src < S; src++) {
    let acc = -Infinity;
    for (let dst = 0; dst < S; dst++) {
      const term = (dst === S - 1) ? 0.0 : -Infinity;
      const val = closure32[src * S + dst] + term;
      if (val !== -Infinity) {
        if (acc === -Infinity) acc = val;
        else {
          const m = Math.max(val, acc);
          acc = m + Math.log(Math.exp(val - m) + Math.exp(acc - m));
        }
      }
    }
    termClosed32[src] = acc;
  }

  if (L === 0) {
    const bp = new Float32Array(S);
    for (let s = 0; s < S; s++) bp[s] = termClosed32[s];
    return { logLikelihood: bp[0], bp };
  }

  // Extract transition slice
  const transSlice32 = new Float32Array(nTok * S * S);
  for (let tok = 0; tok < nTok; tok++) {
    for (let src = 0; src < S; src++) {
      for (let dst = 0; dst < S; dst++) {
        let idx;
        if (isInput) idx = ((tok * nOut + 0) * S + src) * S + dst;
        else idx = ((0 * nOut + tok) * S + src) * S + dst;
        transSlice32[(tok * S + src) * S + dst] = logTransF32[idx];
      }
    }
  }

  // Upload to GPU and build transfer matrices
  const tokensBuffer = createStorageBuffer(device, seq);
  const transSliceBuffer = createStorageBuffer(device, transSlice32);
  const closureBuffer = createStorageBuffer(device, closure32);
  const transfersBuffer = createEmptyBuffer(device, L * S * S * 4);

  const buildShader = loadShader('transfer-build.wgsl');
  const buildPipeline = createPipeline(device, buildShader, { S, N_TOKENS: nTok });
  const buildBindGroup = createBindGroup(device, buildPipeline, 0,
    [tokensBuffer, transSliceBuffer, closureBuffer, transfersBuffer]);

  const enc1 = device.createCommandEncoder();
  const p1 = enc1.beginComputePass();
  p1.setPipeline(buildPipeline);
  p1.setBindGroup(0, buildBindGroup);
  p1.dispatchWorkgroups(L);
  p1.end();
  device.queue.submit([enc1.finish()]);

  // Suffix scan (backward direction)
  const scanShader = loadShader('prefix-scan.wgsl');
  let bufA = transfersBuffer;
  let bufB = createEmptyBuffer(device, L * S * S * 4);
  const nPasses = Math.ceil(Math.log2(L));
  let stride = 1;

  for (let pass = 0; pass < nPasses; pass++) {
    const paramsBuffer = createUniformBuffer(device, new Uint32Array([L, stride]));
    const scanPipeline = createPipeline(device, scanShader, { S, USE_MAX: useMax, IS_BACKWARD: 1 });
    const scanBindGroup = createBindGroup(device, scanPipeline, 0, [bufA, bufB, paramsBuffer]);

    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(scanPipeline);
    p.setBindGroup(0, scanBindGroup);
    p.dispatchWorkgroups(L);
    p.end();
    device.queue.submit([enc.finish()]);

    [bufA, bufB] = [bufB, bufA];
    stride *= 2;
    paramsBuffer.destroy();
  }

  // Read suffix products
  const suffixData = await readBuffer(device, bufA, L * S * S * 4);

  // Extract backward vectors on CPU
  const bp = new Float32Array((L + 1) * S);

  // bp[L] = termClosed
  for (let s = 0; s < S; s++) bp[L * S + s] = termClosed32[s];

  // bp[p] for p = 0..L-1:
  // bwd[p, src] = logsumexp_dst(suffix[p, src, dst] + term_closed[dst])
  for (let p = 0; p < L; p++) {
    const matBase = p * S * S;
    for (let src = 0; src < S; src++) {
      let acc = -Infinity;
      for (let dst = 0; dst < S; dst++) {
        const val = suffixData[matBase + src * S + dst] + termClosed32[dst];
        if (val !== -Infinity) {
          if (acc === -Infinity) acc = val;
          else {
            const m = Math.max(val, acc);
            acc = m + Math.log(Math.exp(val - m) + Math.exp(acc - m));
          }
        }
      }
      bp[p * S + src] = acc;
    }
  }

  const logLikelihood = bp[0];

  // Cleanup
  tokensBuffer.destroy();
  transSliceBuffer.destroy();
  closureBuffer.destroy();
  transfersBuffer.destroy();
  if (bufA !== transfersBuffer) bufA.destroy();
  if (bufB !== transfersBuffer) bufB.destroy();

  return { logLikelihood, bp };
}
