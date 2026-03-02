/**
 * 1D Viterbi algorithm with traceback for PSWM input (CPU fallback).
 *
 * Uses max-plus semiring for generator/recognizer machines.
 */

import { NEG_INF, makeSemiring, logaddexp } from '../internal/logmath.mjs';

const TB_NONE = 0;
const TB_EMIT = 1;
const TB_SILENT = 2;

/**
 * Precompute per-position emission-weighted transition matrices for 1D profile.
 * Uses max instead of logaddexp for max-plus semiring.
 */
function precomputeEmitTransProfile1D(logProfile, nAlpha, transSlice, nTok, S, L) {
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
          const val = emission + trans;
          if (val > acc) acc = val;
        }
        mat[src * S + dst] = acc;
      }
    }
    result[p] = mat;
  }
  return result;
}

/**
 * 1D Viterbi with traceback for PSWM profile.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Float64Array} logProfile - (L * nAlpha) log-probability profile
 * @param {'input'|'output'} direction
 * @param {number} L - profile length
 * @returns {Promise<{score: number, path: Array<{state: number, inputToken: number, outputToken: number}>}>}
 */
export async function viterbi1DProfile(machine, logProfile, direction, L) {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans } = machine;
  const semiring = makeSemiring('maxplus');
  const { reduce } = semiring;

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
  const tbSrc = new Int32Array((L + 1) * S).fill(-1);
  const tbType = new Uint8Array((L + 1) * S).fill(TB_NONE);

  function propagateSilentViterbi(cell, baseIdx) {
    const current = new Float64Array(cell);
    for (let iter = 0; iter < 100; iter++) {
      let changed = false;
      for (let dst = 0; dst < S; dst++) {
        for (let src = 0; src < S; src++) {
          const val = current[src] + silent[src * S + dst];
          if (val > current[dst]) {
            current[dst] = val;
            tbSrc[baseIdx + dst] = src;
            tbType[baseIdx + dst] = TB_SILENT;
            changed = true;
          }
        }
      }
      if (!changed) break;
    }
    return current;
  }

  // Initialize position 0
  dp[0] = 0.0;
  const initCell = propagateSilentViterbi(dp.subarray(0, S), 0);
  for (let s = 0; s < S; s++) dp[s] = initCell[s];

  if (L === 0) {
    return { score: dp[S - 1], path: [] };
  }

  const allTrans = precomputeEmitTransProfile1D(logProfile, nAlpha, transSlice, nTok, S, L);

  for (let p = 0; p < L; p++) {
    const prevBase = p * S;
    const curBase = (p + 1) * S;
    const trans = allTrans[p];

    for (let dst = 0; dst < S; dst++) {
      for (let src = 0; src < S; src++) {
        const val = dp[prevBase + src] + trans[src * S + dst];
        if (val > dp[curBase + dst]) {
          dp[curBase + dst] = val;
          tbSrc[curBase + dst] = src;
          tbType[curBase + dst] = TB_EMIT;
        }
      }
    }

    const closed = propagateSilentViterbi(
      dp.subarray(curBase, curBase + S), curBase
    );
    for (let s = 0; s < S; s++) dp[curBase + s] = closed[s];
  }

  const score = dp[L * S + S - 1];

  // Traceback
  const path = [];
  let p = L, s = S - 1;

  while (p > 0 || s !== 0) {
    const idx = p * S + s;
    const type = tbType[idx];
    const src = tbSrc[idx];

    if (type === TB_NONE || src < 0) break;

    if (type === TB_SILENT) {
      s = src;
    } else if (type === TB_EMIT) {
      if (isInput) {
        path.push({ state: s, inputToken: p, outputToken: 0 });
      } else {
        path.push({ state: s, inputToken: 0, outputToken: p });
      }
      p--;
      s = src;
    }
  }

  path.reverse();
  return { score, path };
}
