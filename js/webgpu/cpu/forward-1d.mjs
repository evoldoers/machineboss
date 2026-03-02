/**
 * 1D Forward algorithm (CPU fallback, Float64Array).
 *
 * For generator (input=null) or recognizer (output=null) machines.
 * Sequential scan: O(L * S^2) time, O(L * S) space.
 *
 * Ported from dp_1d_simple.py.
 */

import { NEG_INF, makeSemiring } from '../internal/logmath.mjs';
import { propagateSilent, emitStepForward } from './silent.mjs';

/**
 * Precompute per-position emission-weighted transition matrices for 1D.
 * For token sequences: at position p, only token seq[p] contributes.
 *
 * @param {Uint32Array} seq - token indices (1-based)
 * @param {Float64Array} transSlice - (nTok, S, S) transition matrices
 * @param {number} nTok
 * @param {number} S
 * @param {function} reduce
 * @returns {Float64Array[]} array of (S*S) Float64Arrays
 */
function precomputeEmitTrans1D(seq, transSlice, nTok, S, reduce) {
  const L = seq.length;
  const result = new Array(L);
  const tmp = new Float64Array(nTok);
  for (let p = 0; p < L; p++) {
    const mat = new Float64Array(S * S);
    for (let src = 0; src < S; src++) {
      for (let dst = 0; dst < S; dst++) {
        for (let tok = 0; tok < nTok; tok++) {
          const emission = (tok === seq[p]) ? 0.0 : NEG_INF;
          tmp[tok] = emission + transSlice[(tok * S + src) * S + dst];
        }
        mat[src * S + dst] = reduce(tmp);
      }
    }
    result[p] = mat;
  }
  return result;
}

/**
 * 1D Forward algorithm.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq - input token indices, or null for generator
 * @param {Uint32Array|null} outputSeq - output token indices, or null for recognizer
 * @param {import('../internal/logmath.mjs').SemiringType} [semiringType='logsumexp']
 * @returns {Promise<number>} log-likelihood
 */
export async function forward1D(machine, inputSeq, outputSeq, semiringType = 'logsumexp') {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans } = machine;
  const semiring = makeSemiring(semiringType);
  const { plus, reduce } = semiring;

  // Determine which dimension is active
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

  // Silent transitions
  const silent = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) silent[i] = logTrans[i];

  // Extract transition slice for the active dimension
  const transSlice = new Float64Array(nTok * S * S);
  for (let tok = 0; tok < nTok; tok++) {
    for (let src = 0; src < S; src++) {
      for (let dst = 0; dst < S; dst++) {
        let idx;
        if (isInput) {
          idx = ((tok * nOut + 0) * S + src) * S + dst;
        } else {
          idx = ((0 * nOut + tok) * S + src) * S + dst;
        }
        transSlice[(tok * S + src) * S + dst] = logTrans[idx];
      }
    }
  }

  if (L === 0) {
    // Only silent transitions from start to end
    const cell = new Float64Array(S).fill(NEG_INF);
    cell[0] = 0.0;
    const closed = propagateSilent(cell, silent, S, plus, reduce);
    return closed[S - 1];
  }

  // Precompute transition matrices
  const allTrans = precomputeEmitTrans1D(seq, transSlice, nTok, S, reduce);

  // DP: fwd[p][s], p = 0..L (position after consuming p tokens)
  // fwd[0] = start state after silent closure
  let prev = new Float64Array(S).fill(NEG_INF);
  prev[0] = 0.0;
  prev = propagateSilent(prev, silent, S, plus, reduce);

  for (let p = 0; p < L; p++) {
    let cell = new Float64Array(S).fill(NEG_INF);
    cell = emitStepForward(cell, prev, allTrans[p], S, plus, reduce);
    cell = propagateSilent(cell, silent, S, plus, reduce);
    prev = cell;
  }

  return prev[S - 1];
}

/**
 * 1D Forward returning full DP grid.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq
 * @param {Uint32Array|null} outputSeq
 * @param {import('../internal/logmath.mjs').SemiringType} [semiringType='logsumexp']
 * @returns {Promise<{logLikelihood: number, dp: Float64Array}>}
 */
export async function forward1DFull(machine, inputSeq, outputSeq, semiringType = 'logsumexp') {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans } = machine;
  const semiring = makeSemiring(semiringType);
  const { plus, reduce } = semiring;

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
  const silent = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) silent[i] = logTrans[i];

  const transSlice = new Float64Array(nTok * S * S);
  for (let tok = 0; tok < nTok; tok++) {
    for (let src = 0; src < S; src++) {
      for (let dst = 0; dst < S; dst++) {
        let idx;
        if (isInput) {
          idx = ((tok * nOut + 0) * S + src) * S + dst;
        } else {
          idx = ((0 * nOut + tok) * S + src) * S + dst;
        }
        transSlice[(tok * S + src) * S + dst] = logTrans[idx];
      }
    }
  }

  // dp[(p) * S + s] for p = 0..L
  const dp = new Float64Array((L + 1) * S).fill(NEG_INF);

  // Initialize position 0
  dp[0] = 0.0;
  const init = propagateSilent(dp.subarray(0, S), silent, S, plus, reduce);
  for (let s = 0; s < S; s++) dp[s] = init[s];

  if (L > 0) {
    const allTrans = precomputeEmitTrans1D(seq, transSlice, nTok, S, reduce);
    for (let p = 0; p < L; p++) {
      const prev = dp.subarray(p * S, (p + 1) * S);
      let cell = new Float64Array(S).fill(NEG_INF);
      cell = emitStepForward(cell, prev, allTrans[p], S, plus, reduce);
      cell = propagateSilent(cell, silent, S, plus, reduce);
      for (let s = 0; s < S; s++) dp[(p + 1) * S + s] = cell[s];
    }
  }

  const ll = dp[L * S + S - 1];
  return { logLikelihood: ll, dp };
}
