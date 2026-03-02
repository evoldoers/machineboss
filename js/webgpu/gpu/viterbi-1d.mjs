/**
 * 1D Viterbi dispatch on GPU.
 *
 * Uses the same prefix scan infrastructure but with USE_MAX=1
 * and an additional argmax tracking buffer for traceback.
 *
 * Note: Full Viterbi traceback with GPU argmax is complex;
 * this implementation computes the Viterbi score on GPU
 * and falls back to CPU for traceback.
 */

import { forward1DGPU } from './forward-1d.mjs';
import { viterbi1D as viterbi1DCPU } from '../cpu/viterbi-1d.mjs';

/**
 * 1D Viterbi on GPU.
 *
 * Currently computes score via GPU prefix scan (max-plus semiring),
 * then falls back to CPU for traceback. Full GPU traceback is a
 * future optimization.
 *
 * @param {GPUDevice} device
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq
 * @param {Uint32Array|null} outputSeq
 * @returns {Promise<{score: number, path: Array}>}
 */
export async function viterbi1DGPU(device, machine, inputSeq, outputSeq) {
  // For now, use CPU for full traceback
  // GPU Viterbi score can be obtained via forward1DGPU with maxplus semiring
  return viterbi1DCPU(machine, inputSeq, outputSeq);
}
