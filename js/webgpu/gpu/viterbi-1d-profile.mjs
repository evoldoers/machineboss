/**
 * 1D Viterbi dispatch for PSWM profile on GPU.
 *
 * Currently falls back to CPU for traceback.
 */

import { viterbi1DProfile as viterbi1DProfileCPU } from '../cpu/viterbi-1d-profile.mjs';

/**
 * 1D Viterbi on GPU for PSWM profile.
 *
 * @param {GPUDevice} device
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Float64Array} logProfile - (L * nAlpha) log-profile
 * @param {'input'|'output'} direction
 * @param {number} L - profile length
 * @returns {Promise<{score: number, path: Array}>}
 */
export async function viterbi1DProfileGPU(device, machine, logProfile, direction, L) {
  // CPU for full traceback; GPU Viterbi score can be added later
  return viterbi1DProfileCPU(machine, logProfile, direction, L);
}
