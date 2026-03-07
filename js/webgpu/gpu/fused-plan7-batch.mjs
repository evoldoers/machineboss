/**
 * Fused Plan7+transducer batch GPU dispatcher.
 *
 * Scores B sequences in parallel on the GPU (one workgroup per sequence).
 * Each thread runs the full fused DP algorithm for its sequence.
 *
 * Model data is packed into a flat Float32Array with a fixed layout
 * matching the WGSL shader's offset functions.
 */

import {
  loadShader, createPipeline, createStorageBuffer,
  createEmptyBuffer, createBindGroup, readBuffer,
} from './pipeline.mjs';

/**
 * Pack FusedPlan7Data into a flat Float32Array for GPU upload.
 *
 * Layout (contiguous f32):
 *   log_m_to_m[K], log_m_to_i[K], log_m_to_d[K],
 *   log_i_to_m[K], log_i_to_i[K], log_d_to_m[K], log_d_to_d[K],
 *   log_match_emit[K*N_AA], log_ins_emit[K*N_AA],
 *   log_b_entry[K], log_null_emit[N_AA],
 *   flanking_weights[8],
 *   td_log_trans[N_IN_TD*N_OUT_TD*S_TD*S_TD],
 *   td_silent[S_TD*S_TD],
 *   aa_to_td_in[N_AA] (u32 bitcast to f32)
 *
 * @param {Object} fm - FusedPlan7Data from buildFusedPlan7
 * @returns {Float32Array}
 */
export function packFusedPlan7ModelData(fm) {
  const { K, n_aa, S_td, n_in_td, n_out_td } = fm;

  const totalSize = 7 * K                           // core transitions
    + 2 * K * n_aa                                   // match + ins emissions
    + K                                              // b_entry
    + n_aa                                           // null_emit
    + 8                                              // flanking weights
    + n_in_td * n_out_td * S_td * S_td               // td_log_trans
    + S_td * S_td                                    // td_silent
    + n_aa;                                          // aa_to_td_in

  const data = new Float32Array(totalSize);
  let offset = 0;

  // Core transitions (7 * K)
  const arrays = [
    fm.log_m_to_m, fm.log_m_to_i, fm.log_m_to_d,
    fm.log_i_to_m, fm.log_i_to_i,
    fm.log_d_to_m, fm.log_d_to_d,
  ];
  for (const arr of arrays) {
    for (let i = 0; i < K; i++) data[offset++] = arr[i];
  }

  // Match and insert emissions (2 * K * N_AA)
  for (let i = 0; i < K * n_aa; i++) data[offset++] = fm.log_match_emit[i];
  for (let i = 0; i < K * n_aa; i++) data[offset++] = fm.log_ins_emit[i];

  // B entry (K)
  for (let i = 0; i < K; i++) data[offset++] = fm.log_b_entry[i];

  // Null emissions (N_AA)
  for (let i = 0; i < n_aa; i++) data[offset++] = fm.log_null_emit[i];

  // Flanking weights (8)
  data[offset++] = fm.log_n_loop;
  data[offset++] = fm.log_n_to_b;
  data[offset++] = fm.log_e_to_cx;
  data[offset++] = fm.log_e_to_jx;
  data[offset++] = fm.log_c_loop;
  data[offset++] = fm.log_c_to_t;
  data[offset++] = fm.log_j_loop;
  data[offset++] = fm.log_j_to_b;

  // Transducer log transitions
  for (let i = 0; i < n_in_td * n_out_td * S_td * S_td; i++) {
    data[offset++] = fm.td_log_trans[i];
  }

  // Transducer silent transitions
  for (let i = 0; i < S_td * S_td; i++) data[offset++] = fm.td_silent[i];

  // aa_to_td_in (u32 bitcast to f32)
  const u32view = new Uint32Array(data.buffer, offset * 4, n_aa);
  for (let i = 0; i < n_aa; i++) u32view[i] = fm.aa_to_td_in[i];
  offset += n_aa;

  return data;
}

/**
 * Pack sequences into a flat Uint32Array with offset table.
 *
 * Layout: [offset_0, offset_1, ..., offset_B, tok_0_0, tok_0_1, ..., tok_{B-1}_last]
 *
 * @param {Uint32Array[]} seqs - Array of tokenized sequences
 * @returns {Uint32Array}
 */
export function packSequences(seqs) {
  const B = seqs.length;
  let totalTokens = 0;
  for (const seq of seqs) totalTokens += seq.length;

  const packed = new Uint32Array(B + 1 + totalTokens);
  let tokenOffset = B + 1;
  for (let i = 0; i < B; i++) {
    packed[i] = tokenOffset;
    for (let j = 0; j < seqs[i].length; j++) {
      packed[tokenOffset++] = seqs[i][j];
    }
  }
  packed[B] = tokenOffset; // end sentinel

  return packed;
}

/**
 * Run fused Plan7+transducer Forward/Viterbi on GPU for a batch of sequences.
 *
 * @param {GPUDevice} device
 * @param {Object} fusedData - FusedPlan7Data from buildFusedPlan7
 * @param {Uint32Array[]} outputSeqs - Array of tokenized output sequences
 * @param {string} [semiringType='logsumexp'] - 'logsumexp' or 'maxplus'
 * @returns {Promise<Float64Array>} B log-likelihoods
 */
export async function fusedPlan7BatchGPU(device, fusedData, outputSeqs, semiringType = 'logsumexp') {
  const B = outputSeqs.length;
  if (B === 0) return new Float64Array(0);

  const { K, n_aa, S_td, n_in_td, n_out_td } = fusedData;
  const useMax = semiringType === 'maxplus' ? 1 : 0;

  // Pack model data
  const modelData = packFusedPlan7ModelData(fusedData);
  const seqData = packSequences(outputSeqs);

  // Workspace: per sequence, need core_m(K*S_td) + core_i(K*S_td) + core_d(K*S_td) + flanking(8*S_td)
  const wsPerSeq = 3 * K * S_td + 8 * S_td;
  const wsSize = B * wsPerSeq;

  // Create GPU buffers
  const modelBuffer = createStorageBuffer(device, modelData);
  const seqBuffer = createStorageBuffer(device, seqData);
  const outputBuffer = createEmptyBuffer(device, B * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const workspaceBuffer = createEmptyBuffer(device, wsSize * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

  // Load and compile shader
  const commonSource = loadShader('fused-plan7-common.wgsl');
  const batchSource = loadShader('fused-plan7-batch.wgsl');
  const shaderSource = commonSource + '\n' + batchSource;

  const pipeline = createPipeline(device, shaderSource, {
    K, S_TD: S_td, N_AA: n_aa,
    N_OUT_TD: n_out_td, N_IN_TD: n_in_td,
    USE_MAX: useMax,
  });

  const bindGroup = createBindGroup(device, pipeline, 0, [
    modelBuffer, seqBuffer, outputBuffer, workspaceBuffer,
  ]);

  // Dispatch B workgroups
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(B);
  pass.end();
  device.queue.submit([encoder.finish()]);

  // Read back results
  const resultF32 = await readBuffer(device, outputBuffer, B * 4);

  // Cleanup
  modelBuffer.destroy();
  seqBuffer.destroy();
  outputBuffer.destroy();
  workspaceBuffer.destroy();

  // Convert to Float64Array
  const result = new Float64Array(B);
  for (let i = 0; i < B; i++) result[i] = resultF32[i];
  return result;
}
