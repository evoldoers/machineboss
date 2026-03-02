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

// CPU fallback imports
import { forward1D, forward1DFull } from './cpu/forward-1d.mjs';
import { forward2D, forward2DFull } from './cpu/forward-2d.mjs';
import { backward1D } from './cpu/backward-1d.mjs';
import { backward2D } from './cpu/backward-2d.mjs';
import { viterbi1D } from './cpu/viterbi-1d.mjs';
import { viterbi2D } from './cpu/viterbi-2d.mjs';
import { posteriors1D, posteriors2D } from './cpu/posteriors.mjs';

// GPU imports (lazy — only loaded when WebGPU is available)
let gpuModules = null;
async function loadGPUModules() {
  if (gpuModules) return gpuModules;
  const [fwd1d, bwd1d, vit1d, fwd2d, bwd2d, vit2d] = await Promise.all([
    import('./gpu/forward-1d.mjs'),
    import('./gpu/backward-1d.mjs'),
    import('./gpu/viterbi-1d.mjs'),
    import('./gpu/forward-2d.mjs'),
    import('./gpu/backward-2d.mjs'),
    import('./gpu/viterbi-2d.mjs'),
  ]);
  gpuModules = { fwd1d, bwd1d, vit1d, fwd2d, bwd2d, vit2d };
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
        throw new Error('WebGPU not available');
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
