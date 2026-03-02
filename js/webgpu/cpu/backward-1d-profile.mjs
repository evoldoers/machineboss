/**
 * 1D Backward algorithm for PSWM (Position-Specific Weight Matrix) input.
 *
 * CPU fallback with Float64Array.
 */

import { NEG_INF, makeSemiring, logaddexp } from '../internal/logmath.mjs';
import { propagateSilentBackward, emitStepBackward } from './silent.mjs';

/**
 * Precompute per-position emission-weighted transition matrices for 1D profile.
 */
function precomputeEmitTransProfile1D(logProfile, nAlpha, transSlice, nTok, S, L, reduce) {
  const result = new Array(L);
  for (let p = 0; p < L; p++) {
    const mat = new Float64Array(S * S);
    for (let src = 0; src < S; src++) {
      for (let dst = 0; dst < S; dst++) {
        let acc = NEG_INF;
        for (let k = 0; k < nAlpha; k++) {
          const tok = k + 1;
          const emission = logProfile[p * nAlpha + k];
          const trans = transSlice[(tok * S + src) * S + dst];
          acc = logaddexp(acc, emission + trans);
        }
        mat[src * S + dst] = acc;
      }
    }
    result[p] = mat;
  }
  return result;
}

/**
 * 1D Backward with PSWM profile returning full grid.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Float64Array} logProfile - (L * nAlpha) log-probability profile
 * @param {'input'|'output'} direction
 * @param {number} L - profile length
 * @param {import('../internal/logmath.mjs').SemiringType} [semiringType='logsumexp']
 * @returns {Promise<{logLikelihood: number, bp: Float64Array}>}
 */
export async function backward1DProfile(machine, logProfile, direction, L, semiringType = 'logsumexp') {
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

  const bp = new Float64Array((L + 1) * S).fill(NEG_INF);

  bp[L * S + S - 1] = 0.0;
  const termCell = propagateSilentBackward(bp.subarray(L * S, (L + 1) * S), silent, S, plus, reduce);
  for (let s = 0; s < S; s++) bp[L * S + s] = termCell[s];

  if (L > 0) {
    const allTrans = precomputeEmitTransProfile1D(logProfile, nAlpha, transSlice, nTok, S, L, reduce);

    for (let p = L - 1; p >= 0; p--) {
      const future = bp.subarray((p + 1) * S, (p + 2) * S);
      let cell = new Float64Array(S).fill(NEG_INF);
      cell = emitStepBackward(cell, future, allTrans[p], S, plus, reduce);
      cell = propagateSilentBackward(cell, silent, S, plus, reduce);
      for (let s = 0; s < S; s++) bp[p * S + s] = cell[s];
    }
  }

  const ll = bp[0];
  return { logLikelihood: ll, bp };
}
