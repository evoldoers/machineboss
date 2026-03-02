/**
 * 1D Forward algorithm for PSWM (Position-Specific Weight Matrix) input.
 *
 * Instead of a tokenized sequence (one token per position), accepts a
 * log-probability profile of shape (L, nAlpha) where nAlpha is the
 * number of emitting symbols (excluding epsilon).
 *
 * CPU fallback with Float64Array.
 */

import { NEG_INF, makeSemiring, logaddexp } from '../internal/logmath.mjs';
import { propagateSilent, emitStepForward } from './silent.mjs';

/**
 * Precompute per-position emission-weighted transition matrices for 1D profile.
 *
 * At each position p, combines all emitting tokens weighted by logProfile:
 *   emitTrans[p][src][dst] = reduce_tok(logProfile[p, tok-1] + transSlice[tok, src, dst])
 *
 * @param {Float64Array} logProfile - (L, nAlpha) log-weights, row-major
 * @param {number} nAlpha - number of emitting symbols (nTok - 1)
 * @param {Float64Array} transSlice - (nTok, S, S) transition matrices
 * @param {number} nTok - total tokens including epsilon
 * @param {number} S - number of states
 * @param {number} L - sequence length
 * @param {function} reduce - semiring reduce
 * @returns {Float64Array[]} array of (S*S) Float64Arrays
 */
function precomputeEmitTransProfile1D(logProfile, nAlpha, transSlice, nTok, S, L, reduce) {
  const result = new Array(L);
  for (let p = 0; p < L; p++) {
    const mat = new Float64Array(S * S);
    for (let src = 0; src < S; src++) {
      for (let dst = 0; dst < S; dst++) {
        // Combine all emitting tokens with profile weights
        let acc = NEG_INF;
        for (let k = 0; k < nAlpha; k++) {
          const tok = k + 1;  // skip epsilon (index 0)
          const emission = logProfile[p * nAlpha + k];
          const trans = transSlice[(tok * S + src) * S + dst];
          const val = emission + trans;
          acc = logaddexp(acc, val);
        }
        mat[src * S + dst] = acc;
      }
    }
    result[p] = mat;
  }
  return result;
}

/**
 * 1D Forward with PSWM profile.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Float64Array} logProfile - (L * nAlpha) log-probability profile
 * @param {'input'|'output'} direction - which alphabet the profile uses
 * @param {number} L - profile length (number of positions)
 * @param {import('../internal/logmath.mjs').SemiringType} [semiringType='logsumexp']
 * @returns {Promise<number>} log-likelihood
 */
export async function forward1DProfile(machine, logProfile, direction, L, semiringType = 'logsumexp') {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans } = machine;
  const semiring = makeSemiring(semiringType);
  const { plus, reduce } = semiring;

  const isInput = direction === 'input';
  const nTok = isInput ? nIn : nOut;
  const nAlpha = nTok - 1;  // exclude epsilon

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
    const cell = new Float64Array(S).fill(NEG_INF);
    cell[0] = 0.0;
    const closed = propagateSilent(cell, silent, S, plus, reduce);
    return closed[S - 1];
  }

  const allTrans = precomputeEmitTransProfile1D(logProfile, nAlpha, transSlice, nTok, S, L, reduce);

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
 * 1D Forward with PSWM returning full DP grid.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Float64Array} logProfile - (L * nAlpha) log-probability profile
 * @param {'input'|'output'} direction
 * @param {number} L - profile length
 * @param {import('../internal/logmath.mjs').SemiringType} [semiringType='logsumexp']
 * @returns {Promise<{logLikelihood: number, dp: Float64Array}>}
 */
export async function forward1DProfileFull(machine, logProfile, direction, L, semiringType = 'logsumexp') {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans } = machine;
  const semiring = makeSemiring(semiringType);
  const { plus, reduce } = semiring;

  const isInput = direction === 'input';
  const nTok = isInput ? nIn : nOut;
  const nAlpha = nTok - 1;

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

  const dp = new Float64Array((L + 1) * S).fill(NEG_INF);

  dp[0] = 0.0;
  const init = propagateSilent(dp.subarray(0, S), silent, S, plus, reduce);
  for (let s = 0; s < S; s++) dp[s] = init[s];

  if (L > 0) {
    const allTrans = precomputeEmitTransProfile1D(logProfile, nAlpha, transSlice, nTok, S, L, reduce);
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
