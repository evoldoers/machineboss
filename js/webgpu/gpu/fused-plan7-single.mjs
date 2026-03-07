/**
 * Fused Plan7+transducer single-sequence GPU dispatcher.
 *
 * One sequence, parallelism within each output position.
 * K threads cooperate per position (one per profile node for emission),
 * with barriers between phases.
 *
 * Falls back to batch kernel (B=1) if K > maxComputeWorkgroupSizeX.
 */

import {
  loadShader, createPipeline, createStorageBuffer,
  createEmptyBuffer, createBindGroup, readBuffer,
} from './pipeline.mjs';
import { packFusedPlan7ModelData } from './fused-plan7-batch.mjs';
import { fusedPlan7BatchGPU } from './fused-plan7-batch.mjs';

/**
 * Next power of 2 >= n.
 */
function nextPow2(n) {
  if (n <= 1) return 1;
  return 1 << (32 - Math.clz32(n - 1));
}

/**
 * Run fused Plan7+transducer Forward/Viterbi on GPU for a single sequence.
 *
 * @param {GPUDevice} device
 * @param {Object} fusedData - FusedPlan7Data from buildFusedPlan7
 * @param {Uint32Array} outputSeq - Tokenized output sequence
 * @param {string} [semiringType='logsumexp'] - 'logsumexp' or 'maxplus'
 * @returns {Promise<number>} log-likelihood
 */
export async function fusedPlan7SingleGPU(device, fusedData, outputSeq, semiringType = 'logsumexp') {
  const { K, n_aa, S_td, n_in_td, n_out_td } = fusedData;
  const useMax = semiringType === 'maxplus' ? 1 : 0;
  const Lo = outputSeq.length;

  // Compute padded workgroup size
  const maxWgSize = device.limits.maxComputeWorkgroupSizeX || 256;
  const kPadded = nextPow2(K);

  // Fall back to batch kernel if K is too large for workgroup
  if (kPadded > maxWgSize) {
    const results = await fusedPlan7BatchGPU(device, fusedData, [outputSeq], semiringType);
    return results[0];
  }

  // Pack model data
  const modelData = packFusedPlan7ModelData(fusedData);

  // Create GPU buffers
  const modelBuffer = createStorageBuffer(device, modelData);
  const seqBuffer = createStorageBuffer(device,
    Lo > 0 ? outputSeq : new Uint32Array([0])); // WebGPU requires non-zero buffer
  const resultBuffer = createEmptyBuffer(device, 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  // Load and compile shader
  const commonSource = loadShader('fused-plan7-common.wgsl');
  const singleSource = loadShader('fused-plan7-single.wgsl');
  const shaderSource = commonSource + '\n' + singleSource;

  const pipeline = createPipeline(device, shaderSource, {
    K, K_PADDED: kPadded, S_TD: S_td, N_AA: n_aa,
    N_OUT_TD: n_out_td, N_IN_TD: n_in_td,
    LO: Lo, USE_MAX: useMax,
  });

  const bindGroup = createBindGroup(device, pipeline, 0, [
    modelBuffer, seqBuffer, resultBuffer,
  ]);

  // Dispatch single workgroup
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  device.queue.submit([encoder.finish()]);

  // Read back result
  const resultF32 = await readBuffer(device, resultBuffer, 4);

  // Cleanup
  modelBuffer.destroy();
  seqBuffer.destroy();
  resultBuffer.destroy();

  return resultF32[0];
}
