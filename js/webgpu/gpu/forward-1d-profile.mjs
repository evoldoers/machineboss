/**
 * 1D Forward dispatch for PSWM profile using transfer-matrix parallel prefix scan.
 *
 * Same pipeline as forward-1d.mjs but uses the profile variant of
 * the transfer-build shader, which combines all emitting tokens
 * weighted by log-profile instead of selecting a single token.
 */

import {
  loadShader, createPipeline, createStorageBuffer,
  createEmptyBuffer, createUniformBuffer, createBindGroup,
  readBuffer,
} from './pipeline.mjs';
import { silentClosure } from '../cpu/silent.mjs';
import { makeSemiring, NEG_INF } from '../internal/logmath.mjs';

/**
 * 1D Forward on GPU for PSWM profile.
 *
 * @param {GPUDevice} device
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Float32Array} logProfile32 - (L * nAlpha) log-profile in Float32
 * @param {'input'|'output'} direction
 * @param {number} L - profile length
 * @param {string} [semiringType='logsumexp']
 * @returns {Promise<number>} log-likelihood
 */
export async function forward1DProfileGPU(device, machine, logProfile32, direction, L, semiringType = 'logsumexp') {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans, logTransF32 } = machine;
  const semiring = makeSemiring(semiringType);
  const useMax = semiringType === 'maxplus' ? 1 : 0;

  const isInput = direction === 'input';
  const nTok = isInput ? nIn : nOut;
  const nAlpha = nTok - 1;

  if (L === 0) {
    const silent = new Float64Array(S * S);
    for (let i = 0; i < S * S; i++) silent[i] = logTrans[i];
    const closure = silentClosure(silent, S, semiring.reduce);
    return closure[S - 1];
  }

  // Silent closure on CPU
  const silent64 = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) silent64[i] = logTrans[i];
  const closure64 = silentClosure(silent64, S, semiring.reduce);
  const closure32 = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++) closure32[i] = closure64[i];

  // Transition slice (f32)
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

  // Upload to GPU
  const profileBuffer = createStorageBuffer(device, logProfile32);
  const transSliceBuffer = createStorageBuffer(device, transSlice32);
  const closureBuffer = createStorageBuffer(device, closure32);
  const transfersBuffer = createEmptyBuffer(device, L * S * S * 4);

  // Build transfer matrices using profile shader
  const buildShader = loadShader('transfer-build-profile.wgsl');
  const buildPipeline = createPipeline(device, buildShader, { S, N_TOKENS: nTok, N_ALPHA: nAlpha });
  const buildBindGroup = createBindGroup(device, buildPipeline, 0,
    [profileBuffer, transSliceBuffer, closureBuffer, transfersBuffer]);

  const encoder1 = device.createCommandEncoder();
  const pass1 = encoder1.beginComputePass();
  pass1.setPipeline(buildPipeline);
  pass1.setBindGroup(0, buildBindGroup);
  pass1.dispatchWorkgroups(L);
  pass1.end();
  device.queue.submit([encoder1.finish()]);

  // Hillis-Steele prefix scan
  const scanShader = loadShader('prefix-scan.wgsl');
  let bufA = transfersBuffer;
  let bufB = createEmptyBuffer(device, L * S * S * 4);
  const nPasses = Math.ceil(Math.log2(L));
  let stride = 1;

  for (let pass = 0; pass < nPasses; pass++) {
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

    [bufA, bufB] = [bufB, bufA];
    stride *= 2;
    paramsBuffer.destroy();
  }

  // Extract forward vector
  const init = new Float32Array(S).fill(-Infinity);
  init[0] = 0.0;
  const initClosed32 = new Float32Array(S);
  for (let dst = 0; dst < S; dst++) {
    let acc = -Infinity;
    for (let src = 0; src < S; src++) {
      const val = init[src] + closure32[src * S + dst];
      if (val > acc) {
        if (acc === -Infinity) acc = val;
        else {
          const m = Math.max(val, acc);
          acc = m + Math.log(Math.exp(val - m) + Math.exp(acc - m));
        }
      }
    }
    initClosed32[dst] = acc;
  }

  const prefixData = await readBuffer(device, bufA, L * S * S * 4);
  const prefixBase = (L - 1) * S * S;
  let logLikelihood = -Infinity;
  for (let dst = 0; dst < S; dst++) {
    let acc = -Infinity;
    for (let src = 0; src < S; src++) {
      const val = initClosed32[src] + prefixData[prefixBase + src * S + dst];
      if (val !== -Infinity) {
        if (acc === -Infinity) acc = val;
        else {
          const m = Math.max(val, acc);
          acc = m + Math.log(Math.exp(val - m) + Math.exp(acc - m));
        }
      }
    }
    if (dst === S - 1) logLikelihood = acc;
  }

  // Cleanup
  profileBuffer.destroy();
  transSliceBuffer.destroy();
  closureBuffer.destroy();
  transfersBuffer.destroy();
  if (bufA !== transfersBuffer) bufA.destroy();
  if (bufB !== transfersBuffer) bufB.destroy();

  return logLikelihood;
}
