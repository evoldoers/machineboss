/**
 * MachineBoss GPU/CPU API.
 *
 * Public API for running Forward, Backward, Viterbi, and posterior
 * computations on Machine Boss transducers, with automatic WebGPU
 * acceleration and CPU fallback.
 *
 * Usage:
 *   import { MachineBoss } from './machineboss-gpu.mjs';
 *   const mb = await MachineBoss.create(machineJSON, { p: 0.9, q: 0.1 });
 *   const ll = await mb.forward(inputTokens, outputTokens);
 *   const { score, path } = await mb.viterbi(inputTokens, outputTokens);
 *   const { logLikelihood, posteriors } = await mb.posteriors(inputTokens, outputTokens);
 *   mb.destroy();
 */

import { prepareMachine, tokenize } from './internal/machine-prep.mjs';
import { detectBackend } from './internal/detect-backend.mjs';
import { parseHmmer } from './internal/hmmer-parse.mjs';
import { buildFusedPlan7, fusedPlan7Forward, fusedPlan7Viterbi } from './cpu/fused-plan7.mjs';

// CPU fallback imports
import { forward1D, forward1DFull } from './cpu/forward-1d.mjs';
import { forward2D, forward2DFull } from './cpu/forward-2d.mjs';
import { backward1D } from './cpu/backward-1d.mjs';
import { backward2D } from './cpu/backward-2d.mjs';
import { viterbi1D } from './cpu/viterbi-1d.mjs';
import { viterbi2D } from './cpu/viterbi-2d.mjs';
import { posteriors1D, posteriors2D } from './cpu/posteriors.mjs';

// PSWM profile imports (CPU)
import { forward1DProfile, forward1DProfileFull } from './cpu/forward-1d-profile.mjs';
import { backward1DProfile } from './cpu/backward-1d-profile.mjs';
import { viterbi1DProfile } from './cpu/viterbi-1d-profile.mjs';

// GPU imports (lazy — only loaded when WebGPU is available)
let gpuModules = null;
async function loadGPUModules() {
  if (gpuModules) return gpuModules;
  const [fwd1d, bwd1d, vit1d, fwd2d, bwd2d, vit2d, fwdProf1d, bwdProf1d, vitProf1d] = await Promise.all([
    import('./gpu/forward-1d.mjs'),
    import('./gpu/backward-1d.mjs'),
    import('./gpu/viterbi-1d.mjs'),
    import('./gpu/forward-2d.mjs'),
    import('./gpu/backward-2d.mjs'),
    import('./gpu/viterbi-2d.mjs'),
    import('./gpu/forward-1d-profile.mjs'),
    import('./gpu/backward-1d-profile.mjs'),
    import('./gpu/viterbi-1d-profile.mjs'),
  ]);
  gpuModules = { fwd1d, bwd1d, vit1d, fwd2d, bwd2d, vit2d, fwdProf1d, bwdProf1d, vitProf1d };
  return gpuModules;
}

/**
 * @typedef {'webgpu'|'cpu'} Backend
 */

export class MachineBoss {
  /**
   * @private - use MachineBoss.create() instead
   */
  constructor(machine, backend, device) {
    /** @type {import('./internal/machine-prep.mjs').PreparedMachine} */
    this._machine = machine;
    /** @type {Backend} */
    this.backend = backend;
    /** @type {GPUDevice|null} */
    this._device = device;
  }

  /**
   * Create a MachineBoss instance.
   *
   * @param {Object} machineJSON - Machine Boss JSON object
   * @param {Object<string,number>} [params={}] - Parameter values
   * @param {Object} [options={}]
   * @param {'auto'|'webgpu'|'cpu'} [options.backend='auto'] - Backend preference
   * @returns {Promise<MachineBoss>}
   */
  static async create(machineJSON, params = {}, options = {}) {
    const machine = prepareMachine(machineJSON, params);

    let backend, device;
    const pref = options.backend || 'auto';

    if (pref === 'cpu') {
      backend = 'cpu';
      device = null;
    } else if (pref === 'webgpu') {
      const detected = await detectBackend();
      if (detected.backend !== 'webgpu') {
        throw new Error('WebGPU is not available in this environment. Use { backend: "cpu" } or "auto" to fall back to CPU.');
      }
      backend = 'webgpu';
      device = detected.device;
    } else {
      // auto
      const detected = await detectBackend();
      backend = detected.backend;
      device = detected.device;
    }

    return new MachineBoss(machine, backend, device);
  }

  /**
   * Number of states in the machine.
   * @returns {number}
   */
  get nStates() {
    return this._machine.nStates;
  }

  /**
   * Input token alphabet (index 0 = null/epsilon).
   * @returns {string[]}
   */
  get inputAlphabet() {
    return this._machine.inputAlphabet;
  }

  /**
   * Output token alphabet (index 0 = null/epsilon).
   * @returns {string[]}
   */
  get outputAlphabet() {
    return this._machine.outputAlphabet;
  }

  /**
   * Determine if this is a 1D or 2D computation.
   * @param {Uint32Array|null} inputTokens
   * @param {Uint32Array|null} outputTokens
   * @returns {boolean} true if 1D (one dimension is null)
   */
  _is1D(inputTokens, outputTokens) {
    return inputTokens === null || inputTokens === undefined ||
           outputTokens === null || outputTokens === undefined;
  }

  /**
   * Compute Forward log-likelihood.
   *
   * @param {Uint32Array|null} inputTokens - 1-based input token indices, or null
   * @param {Uint32Array|null} outputTokens - 1-based output token indices, or null
   * @returns {Promise<number>} log P(input, output | machine)
   */
  async forward(inputTokens, outputTokens) {
    if (this.backend === 'webgpu') {
      try {
        const gpu = await loadGPUModules();
        if (this._is1D(inputTokens, outputTokens)) {
          return gpu.fwd1d.forward1DGPU(this._device, this._machine, inputTokens, outputTokens);
        }
        return gpu.fwd2d.forward2DGPU(this._device, this._machine, inputTokens, outputTokens);
      } catch (e) {
        // Fall back to CPU on GPU error
      }
    }

    if (this._is1D(inputTokens, outputTokens)) {
      return forward1D(this._machine, inputTokens, outputTokens);
    }
    return forward2D(this._machine, inputTokens, outputTokens);
  }

  /**
   * Compute Viterbi best path and score.
   *
   * @param {Uint32Array|null} inputTokens
   * @param {Uint32Array|null} outputTokens
   * @returns {Promise<{score: number, path: Array<{state: number, inputToken: number, outputToken: number}>}>}
   */
  async viterbi(inputTokens, outputTokens) {
    if (this.backend === 'webgpu') {
      try {
        const gpu = await loadGPUModules();
        if (this._is1D(inputTokens, outputTokens)) {
          return gpu.vit1d.viterbi1DGPU(this._device, this._machine, inputTokens, outputTokens);
        }
        return gpu.vit2d.viterbi2DGPU(this._device, this._machine, inputTokens, outputTokens);
      } catch (e) {
        // Fall back to CPU
      }
    }

    if (this._is1D(inputTokens, outputTokens)) {
      return viterbi1D(this._machine, inputTokens, outputTokens);
    }
    return viterbi2D(this._machine, inputTokens, outputTokens);
  }

  /**
   * Compute Forward-Backward posteriors.
   *
   * @param {Uint32Array|null} inputTokens
   * @param {Uint32Array|null} outputTokens
   * @returns {Promise<{logLikelihood: number, posteriors: Float32Array}>}
   *   posteriors shape: 1D = (L+1)*S, 2D = (Li+1)*(Lo+1)*S
   */
  async posteriors(inputTokens, outputTokens) {
    // Posteriors always use CPU for now (requires both forward and backward passes)
    // GPU posteriors can be added as an optimization later
    if (this._is1D(inputTokens, outputTokens)) {
      return posteriors1D(this._machine, inputTokens, outputTokens);
    }
    return posteriors2D(this._machine, inputTokens, outputTokens);
  }

  /**
   * Tokenize a symbol sequence using the machine's alphabet.
   *
   * @param {string|string[]} seq - Sequence of symbols
   * @param {'input'|'output'} direction
   * @returns {Uint32Array} 1-based token indices
   */
  tokenize(seq, direction) {
    const alphabet = direction === 'input' ? this.inputAlphabet : this.outputAlphabet;
    return tokenize(seq, alphabet);
  }

  /**
   * Get the number of emitting symbols for a direction.
   * This is the alphabet size excluding the epsilon token.
   *
   * @param {'input'|'output'} direction
   * @returns {number}
   */
  nAlpha(direction) {
    const nTok = direction === 'input' ? this._machine.nInputTokens : this._machine.nOutputTokens;
    return nTok - 1;  // exclude epsilon
  }

  /**
   * Create a log-profile from probability values.
   * Utility to convert a flat probability array to log-space.
   *
   * @param {Float64Array|Float32Array|number[]} probs - (L * nAlpha) probabilities
   * @returns {Float64Array} log-probabilities
   */
  static logProfile(probs) {
    const result = new Float64Array(probs.length);
    for (let i = 0; i < probs.length; i++) {
      result[i] = probs[i] > 0 ? Math.log(probs[i]) : -Infinity;
    }
    return result;
  }

  /**
   * Compute Forward log-likelihood with a PSWM profile.
   *
   * The profile is a flat array of log-weights, shape (L * nAlpha),
   * where nAlpha = mb.nAlpha(direction) is the number of emitting symbols.
   * profile[p * nAlpha + k] = log P(symbol k at position p).
   *
   * @param {Float64Array} logProfile - (L * nAlpha) log-probability profile
   * @param {'input'|'output'} direction - which alphabet the profile uses
   * @returns {Promise<number>} log P(profile | machine)
   */
  async forwardProfile(logProfile, direction) {
    const nAlpha = this.nAlpha(direction);
    const L = logProfile.length / nAlpha;

    if (this.backend === 'webgpu') {
      try {
        const gpu = await loadGPUModules();
        const logProfile32 = new Float32Array(logProfile);
        return gpu.fwdProf1d.forward1DProfileGPU(this._device, this._machine, logProfile32, direction, L);
      } catch (e) {
        // Fall back to CPU
      }
    }

    return forward1DProfile(this._machine, logProfile, direction, L);
  }

  /**
   * Compute Viterbi best path with a PSWM profile.
   *
   * @param {Float64Array} logProfile - (L * nAlpha) log-probability profile
   * @param {'input'|'output'} direction
   * @returns {Promise<{score: number, path: Array<{state: number, inputToken: number, outputToken: number}>}>}
   */
  async viterbiProfile(logProfile, direction) {
    const nAlpha = this.nAlpha(direction);
    const L = logProfile.length / nAlpha;

    if (this.backend === 'webgpu') {
      try {
        const gpu = await loadGPUModules();
        return gpu.vitProf1d.viterbi1DProfileGPU(this._device, this._machine, logProfile, direction, L);
      } catch (e) {
        // Fall back to CPU
      }
    }

    return viterbi1DProfile(this._machine, logProfile, direction, L);
  }

  /**
   * Compute Forward-Backward posteriors with a PSWM profile.
   *
   * @param {Float64Array} logProfile - (L * nAlpha) log-probability profile
   * @param {'input'|'output'} direction
   * @returns {Promise<{logLikelihood: number, posteriors: Float32Array}>}
   *   posteriors shape: (L+1) * S
   */
  async posteriorsProfile(logProfile, direction) {
    const nAlpha = this.nAlpha(direction);
    const L = logProfile.length / nAlpha;
    const S = this._machine.nStates;

    const [fwdResult, bwdResult] = await Promise.all([
      forward1DProfileFull(this._machine, logProfile, direction, L),
      backward1DProfile(this._machine, logProfile, direction, L),
    ]);

    const { logLikelihood, dp: fwd } = fwdResult;
    const { bp: bwd } = bwdResult;

    const size = (L + 1) * S;
    const posteriors = new Float32Array(size);

    for (let p = 0; p <= L; p++) {
      for (let s = 0; s < S; s++) {
        const idx = p * S + s;
        const logPost = fwd[idx] + bwd[idx] - logLikelihood;
        posteriors[idx] = logPost === -Infinity ? 0 : Math.exp(logPost);
      }
    }

    return { logLikelihood, posteriors };
  }

  /**
   * Create a MachineBoss instance for fused Plan7+transducer DP.
   *
   * @param {string} hmmerText - HMMER3 profile text
   * @param {Object} transducerJSON - Transducer machine JSON object
   * @param {Object<string,number>} [params={}] - Transducer parameter values
   * @param {Object} [options={}]
   * @param {boolean} [options.multihit=false] - Enable multi-hit mode
   * @param {number} [options.L=400] - Expected sequence length for flanking states
   * @returns {Promise<MachineBoss>}
   */
  static async createFusedPlan7(hmmerText, transducerJSON, params = {}, options = {}) {
    const hmmerModel = parseHmmer(hmmerText);
    const prepared = prepareMachine(transducerJSON, params);
    const fused = buildFusedPlan7(hmmerModel, prepared, {
      multihit: options.multihit || false,
      L: options.L || 400,
    });
    const instance = new MachineBoss(prepared, 'cpu', null);
    instance._fusedPlan7 = fused;
    return instance;
  }

  /**
   * Compute fused Plan7+transducer Forward log-likelihood.
   *
   * @param {Uint32Array} outputTokens - 1-based output token indices
   * @returns {Promise<number>}
   */
  async fusedForward(outputTokens) {
    if (!this._fusedPlan7) throw new Error('Not a fused Plan7 instance; use createFusedPlan7()');
    return fusedPlan7Forward(this._fusedPlan7, outputTokens);
  }

  /**
   * Compute fused Plan7+transducer Viterbi score.
   *
   * @param {Uint32Array} outputTokens - 1-based output token indices
   * @returns {Promise<number>}
   */
  async fusedViterbi(outputTokens) {
    if (!this._fusedPlan7) throw new Error('Not a fused Plan7 instance; use createFusedPlan7()');
    return fusedPlan7Viterbi(this._fusedPlan7, outputTokens);
  }

  /**
   * Release GPU resources.
   */
  destroy() {
    if (this._device) {
      this._device.destroy();
      this._device = null;
    }
    this.backend = 'cpu';
  }
}
