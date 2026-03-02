/**
 * 1D Backward dispatch for PSWM profile using transfer-matrix parallel suffix scan.
 */

import {
  loadShader, createPipeline, createStorageBuffer,
  createEmptyBuffer, createUniformBuffer, createBindGroup,
  readBuffer,
} from './pipeline.mjs';
import { silentClosure } from '../cpu/silent.mjs';
import { makeSemiring, NEG_INF } from '../internal/logmath.mjs';

/**
 * 1D Backward on GPU for PSWM profile.
 *
 * @param {GPUDevice} device
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Float32Array} logProfile32 - (L * nAlpha) log-profile in Float32
 * @param {'input'|'output'} direction
 * @param {number} L - profile length
 * @param {string} [semiringType='logsumexp']
 * @returns {Promise<{logLikelihood: number, bp: Float32Array}>}
 */
export async function backward1DProfileGPU(device, machine, logProfile32, direction, L, semiringType = 'logsumexp') {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans, logTransF32 } = machine;
  const semiring = makeSemiring(semiringType);
  const useMax = semiringType === 'maxplus' ? 1 : 0;

  const isInput = direction === 'input';
  const nTok = isInput ? nIn : nOut;
  const nAlpha = nTok - 1;

  // Silent closure on CPU
  const silent64 = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) silent64[i] = logTrans[i];
  const closure64 = silentClosure(silent64, S, semiring.reduce);
  const closure32 = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++) closure32[i] = closure64[i];

  // Terminal vector
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

  // Transition slice
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

  // Upload and build transfer matrices
  const profileBuffer = createStorageBuffer(device, logProfile32);
  const transSliceBuffer = createStorageBuffer(device, transSlice32);
  const closureBuffer = createStorageBuffer(device, closure32);
  const transfersBuffer = createEmptyBuffer(device, L * S * S * 4);

  const buildShader = loadShader('transfer-build-profile.wgsl');
  const buildPipeline = createPipeline(device, buildShader, { S, N_TOKENS: nTok, N_ALPHA: nAlpha });
  const buildBindGroup = createBindGroup(device, buildPipeline, 0,
    [profileBuffer, transSliceBuffer, closureBuffer, transfersBuffer]);

  const enc1 = device.createCommandEncoder();
  const p1 = enc1.beginComputePass();
  p1.setPipeline(buildPipeline);
  p1.setBindGroup(0, buildBindGroup);
  p1.dispatchWorkgroups(L);
  p1.end();
  device.queue.submit([enc1.finish()]);

  // Suffix scan (backward)
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

  const suffixData = await readBuffer(device, bufA, L * S * S * 4);

  // Extract backward vectors
  const bp = new Float32Array((L + 1) * S);
  for (let s = 0; s < S; s++) bp[L * S + s] = termClosed32[s];

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
  profileBuffer.destroy();
  transSliceBuffer.destroy();
  closureBuffer.destroy();
  transfersBuffer.destroy();
  if (bufA !== transfersBuffer) bufA.destroy();
  if (bufB !== transfersBuffer) bufB.destroy();

  return { logLikelihood, bp };
}
