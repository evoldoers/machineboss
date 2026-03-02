/**
 * 2D Forward algorithm (CPU fallback, Float64Array).
 *
 * Computes log P(input, output | machine) for a transducer using the
 * standard row-major nested-loop DP, ported from dp_2d_optimal.py.
 *
 * Complexity: O(Li * Lo * S^2) time, O(Li * Lo * S) space.
 */

import { NEG_INF, makeSemiring } from '../internal/logmath.mjs';
import { propagateSilent, emitStepForward } from './silent.mjs';

/**
 * Precompute per-position emission-weighted transition matrices.
 * For each position p, reduces over token dimension:
 *   result[p][src][dst] = reduce_tok(emission[p][tok] + transSlice[tok][src][dst])
 *
 * @param {Uint32Array} seq - token indices (1-based)
 * @param {Float64Array} transSlice - (nTok, S, S) transition matrices for relevant dimension
 * @param {number} nTok - number of tokens including null
 * @param {number} S
 * @param {function} reduce
 * @returns {Float64Array[]} array of (S*S) Float64Arrays, one per position
 */
function precomputeEmitTrans(seq, transSlice, nTok, S, reduce) {
  const L = seq.length;
  const result = new Array(L);
  const tmp = new Float64Array(nTok);
  for (let p = 0; p < L; p++) {
    const mat = new Float64Array(S * S);
    for (let src = 0; src < S; src++) {
      for (let dst = 0; dst < S; dst++) {
        // Reduce over all tokens: only the actual token at this position contributes
        // For token sequences (not PSWMs), emission weight is 0 for the actual token, NEG_INF for others
        for (let tok = 0; tok < nTok; tok++) {
          // emission weight: 0 for the token at this position, NEG_INF otherwise
          // Skip null token (tok=0)
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
 * Precompute match transition matrices for all (input_pos, output_pos) pairs.
 *
 * @param {Uint32Array} inputSeq
 * @param {Uint32Array} outputSeq
 * @param {Float64Array} logTrans - full (nIn, nOut, S, S) tensor
 * @param {number} nIn
 * @param {number} nOut
 * @param {number} S
 * @param {function} reduce
 * @returns {Float64Array[]} flat array of (S*S) matrices, indexed as [i * Lo + o]
 */
function precomputeMatch(inputSeq, outputSeq, logTrans, nIn, nOut, S, reduce) {
  const Li = inputSeq.length;
  const Lo = outputSeq.length;
  const result = new Array(Li * Lo);
  for (let i = 0; i < Li; i++) {
    const inTok = inputSeq[i];
    for (let o = 0; o < Lo; o++) {
      const outTok = outputSeq[o];
      const mat = new Float64Array(S * S);
      for (let src = 0; src < S; src++) {
        for (let dst = 0; dst < S; dst++) {
          mat[src * S + dst] = logTrans[((inTok * nOut + outTok) * S + src) * S + dst];
        }
      }
      result[i * Lo + o] = mat;
    }
  }
  return result;
}

/**
 * Extract a submatrix transSlice[tok][src][dst] from the full tensor.
 * For insertions: transSlice = logTrans[inTok, 0, :, :] for all inTok
 * For deletions: transSlice = logTrans[0, outTok, :, :] for all outTok
 *
 * @param {Float64Array} logTrans
 * @param {number} nIn
 * @param {number} nOut
 * @param {number} S
 * @param {boolean} isInput - true for input (insert) slice, false for output (delete) slice
 * @returns {Float64Array} (nTok, S, S) where nTok = nIn or nOut
 */
function extractTransSlice(logTrans, nIn, nOut, S, isInput) {
  const nTok = isInput ? nIn : nOut;
  const slice = new Float64Array(nTok * S * S);
  for (let tok = 0; tok < nTok; tok++) {
    for (let src = 0; src < S; src++) {
      for (let dst = 0; dst < S; dst++) {
        let idx;
        if (isInput) {
          // logTrans[tok, 0, src, dst]
          idx = ((tok * nOut + 0) * S + src) * S + dst;
        } else {
          // logTrans[0, tok, src, dst]
          idx = ((0 * nOut + tok) * S + src) * S + dst;
        }
        slice[(tok * S + src) * S + dst] = logTrans[idx];
      }
    }
  }
  return slice;
}

/**
 * 2D Forward algorithm.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq - input token indices (1-based), or null
 * @param {Uint32Array|null} outputSeq - output token indices (1-based), or null
 * @param {import('../internal/logmath.mjs').SemiringType} [semiringType='logsumexp']
 * @returns {Promise<number>} log-likelihood (or Viterbi score)
 */
export async function forward2D(machine, inputSeq, outputSeq, semiringType = 'logsumexp') {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans } = machine;
  const semiring = makeSemiring(semiringType);
  const { plus, reduce } = semiring;

  const Li = inputSeq ? inputSeq.length : 0;
  const Lo = outputSeq ? outputSeq.length : 0;

  // Silent transitions: logTrans[0][0]
  const silent = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) silent[i] = logTrans[i];

  // DP grid: dp[(i * (Lo+1) + o) * S + s]
  const dpSize = (Li + 1) * (Lo + 1) * S;
  const dp = new Float64Array(dpSize).fill(NEG_INF);

  function getCell(i, o) {
    const base = (i * (Lo + 1) + o) * S;
    return dp.subarray(base, base + S);
  }

  function setCell(i, o, values) {
    const base = (i * (Lo + 1) + o) * S;
    for (let s = 0; s < S; s++) dp[base + s] = values[s];
  }

  // Initialize (0,0): start state = 0
  dp[0] = 0.0;
  const init = propagateSilent(getCell(0, 0), silent, S, plus, reduce);
  setCell(0, 0, init);

  if (Li + Lo === 0) {
    return dp[S - 1];
  }

  // Precompute transition matrices
  const insSlice = Li > 0 ? extractTransSlice(logTrans, nIn, nOut, S, true) : null;
  const delSlice = Lo > 0 ? extractTransSlice(logTrans, nIn, nOut, S, false) : null;
  const allIns = Li > 0 ? precomputeEmitTrans(inputSeq, insSlice, nIn, S, reduce) : null;
  const allDel = Lo > 0 ? precomputeEmitTrans(outputSeq, delSlice, nOut, S, reduce) : null;
  const allMatch = (Li > 0 && Lo > 0) ? precomputeMatch(inputSeq, outputSeq, logTrans, nIn, nOut, S, reduce) : null;

  const negInfCell = new Float64Array(S).fill(NEG_INF);

  // Fill DP grid using anti-diagonal wavefront (matches 2D optimal structure)
  for (let d = 1; d <= Li + Lo; d++) {
    const iMin = Math.max(0, d - Lo);
    const iMax = Math.min(Li, d);
    for (let i = iMin; i <= iMax; i++) {
      const o = d - i;
      if (o < 0 || o > Lo) continue;
      if (i === 0 && o === 0) continue;

      let cell = new Float64Array(S).fill(NEG_INF);

      // Match from (i-1, o-1)
      if (i > 0 && o > 0) {
        const mt = allMatch[(i - 1) * Lo + (o - 1)];
        cell = emitStepForward(cell, getCell(i - 1, o - 1), mt, S, plus, reduce);
      }

      // Insert from (i-1, o)
      if (i > 0) {
        const it = allIns[i - 1];
        cell = emitStepForward(cell, getCell(i - 1, o), it, S, plus, reduce);
      }

      // Delete from (i, o-1)
      if (o > 0) {
        const dt = allDel[o - 1];
        cell = emitStepForward(cell, getCell(i, o - 1), dt, S, plus, reduce);
      }

      // Propagate silent transitions
      cell = propagateSilent(cell, silent, S, plus, reduce);
      setCell(i, o, cell);
    }
  }

  return dp[((Li) * (Lo + 1) + Lo) * S + S - 1];
}

/**
 * 2D Forward returning the full DP grid.
 * Used internally by posteriors and other algorithms that need the grid.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq
 * @param {Uint32Array|null} outputSeq
 * @param {import('../internal/logmath.mjs').SemiringType} [semiringType='logsumexp']
 * @returns {Promise<{logLikelihood: number, dp: Float64Array}>}
 */
export async function forward2DFull(machine, inputSeq, outputSeq, semiringType = 'logsumexp') {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans } = machine;
  const semiring = makeSemiring(semiringType);
  const { plus, reduce } = semiring;

  const Li = inputSeq ? inputSeq.length : 0;
  const Lo = outputSeq ? outputSeq.length : 0;

  const silent = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) silent[i] = logTrans[i];

  const dpSize = (Li + 1) * (Lo + 1) * S;
  const dp = new Float64Array(dpSize).fill(NEG_INF);

  function getCell(i, o) {
    const base = (i * (Lo + 1) + o) * S;
    return dp.subarray(base, base + S);
  }

  function setCell(i, o, values) {
    const base = (i * (Lo + 1) + o) * S;
    for (let s = 0; s < S; s++) dp[base + s] = values[s];
  }

  dp[0] = 0.0;
  const init = propagateSilent(getCell(0, 0), silent, S, plus, reduce);
  setCell(0, 0, init);

  if (Li + Lo > 0) {
    const insSlice = Li > 0 ? extractTransSlice(logTrans, nIn, nOut, S, true) : null;
    const delSlice = Lo > 0 ? extractTransSlice(logTrans, nIn, nOut, S, false) : null;
    const allIns = Li > 0 ? precomputeEmitTrans(inputSeq, insSlice, nIn, S, reduce) : null;
    const allDel = Lo > 0 ? precomputeEmitTrans(outputSeq, delSlice, nOut, S, reduce) : null;
    const allMatch = (Li > 0 && Lo > 0) ? precomputeMatch(inputSeq, outputSeq, logTrans, nIn, nOut, S, reduce) : null;

    for (let d = 1; d <= Li + Lo; d++) {
      const iMin = Math.max(0, d - Lo);
      const iMax = Math.min(Li, d);
      for (let i = iMin; i <= iMax; i++) {
        const o = d - i;
        if (o < 0 || o > Lo) continue;
        if (i === 0 && o === 0) continue;

        let cell = new Float64Array(S).fill(NEG_INF);

        if (i > 0 && o > 0) {
          const mt = allMatch[(i - 1) * Lo + (o - 1)];
          cell = emitStepForward(cell, getCell(i - 1, o - 1), mt, S, plus, reduce);
        }
        if (i > 0) {
          const it = allIns[i - 1];
          cell = emitStepForward(cell, getCell(i - 1, o), it, S, plus, reduce);
        }
        if (o > 0) {
          const dt = allDel[o - 1];
          cell = emitStepForward(cell, getCell(i, o - 1), dt, S, plus, reduce);
        }

        cell = propagateSilent(cell, silent, S, plus, reduce);
        setCell(i, o, cell);
      }
    }
  }

  const ll = dp[((Li) * (Lo + 1) + Lo) * S + S - 1];
  return { logLikelihood: ll, dp };
}
