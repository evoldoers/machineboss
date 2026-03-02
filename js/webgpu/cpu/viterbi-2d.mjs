/**
 * 2D Viterbi algorithm with traceback (CPU fallback).
 *
 * Uses max-plus semiring to find the highest-scoring path through
 * the transducer, then traces back to recover the alignment.
 */

import { NEG_INF, makeSemiring } from '../internal/logmath.mjs';
import { propagateSilent, emitStepForward } from './silent.mjs';

/**
 * @typedef {Object} ViterbiResult
 * @property {number} score - Viterbi log-score
 * @property {Array<{state: number, inputToken: number, outputToken: number}>} path - alignment path
 */

function precomputeEmitTrans(seq, transSlice, nTok, S, reduce) {
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

function extractTransSlice(logTrans, nIn, nOut, S, isInput) {
  const nTok = isInput ? nIn : nOut;
  const slice = new Float64Array(nTok * S * S);
  for (let tok = 0; tok < nTok; tok++) {
    for (let src = 0; src < S; src++) {
      for (let dst = 0; dst < S; dst++) {
        let idx;
        if (isInput) {
          idx = ((tok * nOut + 0) * S + src) * S + dst;
        } else {
          idx = ((0 * nOut + tok) * S + src) * S + dst;
        }
        slice[(tok * S + src) * S + dst] = logTrans[idx];
      }
    }
  }
  return slice;
}

// Traceback type constants
const TB_NONE = 0;
const TB_MATCH = 1;
const TB_INSERT = 2;
const TB_DELETE = 3;
const TB_SILENT = 4;

/**
 * Viterbi forward pass with argmax tracking.
 * For each cell (i, o, dst), stores the best source state and move type.
 */
function viterbiForwardWithTraceback(machine, inputSeq, outputSeq) {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans } = machine;
  const semiring = makeSemiring('maxplus');
  const { plus, reduce, argreduce } = semiring;

  const Li = inputSeq ? inputSeq.length : 0;
  const Lo = outputSeq ? outputSeq.length : 0;

  const silent = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) silent[i] = logTrans[i];

  const dpSize = (Li + 1) * (Lo + 1) * S;
  const dp = new Float64Array(dpSize).fill(NEG_INF);

  // Traceback: for each cell, store (srcState, moveType, srcI, srcO)
  // Encoded as: tb[(i*(Lo+1)+o)*S + dst] = { srcState, moveType }
  const tbSrc = new Int32Array(dpSize).fill(-1);
  const tbType = new Uint8Array(dpSize).fill(TB_NONE);

  function getCell(i, o) {
    const base = (i * (Lo + 1) + o) * S;
    return dp.subarray(base, base + S);
  }

  function setCellWithTraceback(i, o, values, sources, types) {
    const base = (i * (Lo + 1) + o) * S;
    for (let s = 0; s < S; s++) {
      dp[base + s] = values[s];
      tbSrc[base + s] = sources[s];
      tbType[base + s] = types[s];
    }
  }

  /**
   * Propagate silent transitions with argmax tracking.
   */
  function propagateSilentViterbi(cell, base) {
    const current = new Float64Array(cell);
    const sources = new Int32Array(S).fill(-1);
    const types = new Uint8Array(S).fill(TB_NONE);

    for (let iter = 0; iter < 100; iter++) {
      let changed = false;
      for (let dst = 0; dst < S; dst++) {
        for (let src = 0; src < S; src++) {
          const val = current[src] + silent[src * S + dst];
          if (val > current[dst]) {
            current[dst] = val;
            sources[dst] = src;
            types[dst] = TB_SILENT;
            changed = true;
          }
        }
      }
      if (!changed) break;
    }

    // Update traceback for states that changed
    for (let s = 0; s < S; s++) {
      if (sources[s] >= 0) {
        tbSrc[base + s] = sources[s];
        tbType[base + s] = types[s];
      }
    }

    return current;
  }

  /**
   * Emit step with argmax tracking.
   * Returns updated cell values and populates traceback entries.
   */
  function emitStepViterbi(cell, prev, trans, moveType, base) {
    const result = new Float64Array(cell);
    for (let dst = 0; dst < S; dst++) {
      for (let src = 0; src < S; src++) {
        const val = prev[src] + trans[src * S + dst];
        if (val > result[dst]) {
          result[dst] = val;
          tbSrc[base + dst] = src;
          tbType[base + dst] = moveType;
        }
      }
    }
    return result;
  }

  // Initialize (0,0)
  dp[0] = 0.0;
  const initBase = 0;
  const initCell = propagateSilentViterbi(getCell(0, 0), initBase);
  for (let s = 0; s < S; s++) dp[s] = initCell[s];

  if (Li + Lo === 0) {
    return { dp, tbSrc, tbType, score: dp[S - 1] };
  }

  // Precompute
  const insSlice = Li > 0 ? extractTransSlice(logTrans, nIn, nOut, S, true) : null;
  const delSlice = Lo > 0 ? extractTransSlice(logTrans, nIn, nOut, S, false) : null;
  const allIns = Li > 0 ? precomputeEmitTrans(inputSeq, insSlice, nIn, S, reduce) : null;
  const allDel = Lo > 0 ? precomputeEmitTrans(outputSeq, delSlice, nOut, S, reduce) : null;
  const allMatch = (Li > 0 && Lo > 0) ? precomputeMatch(inputSeq, outputSeq, logTrans, nIn, nOut, S, reduce) : null;

  // Fill DP grid
  for (let d = 1; d <= Li + Lo; d++) {
    const iMin = Math.max(0, d - Lo);
    const iMax = Math.min(Li, d);
    for (let i = iMin; i <= iMax; i++) {
      const o = d - i;
      if (o < 0 || o > Lo) continue;
      if (i === 0 && o === 0) continue;

      const base = (i * (Lo + 1) + o) * S;
      let cell = new Float64Array(S).fill(NEG_INF);

      // Match from (i-1, o-1)
      if (i > 0 && o > 0) {
        const mt = allMatch[(i - 1) * Lo + (o - 1)];
        cell = emitStepViterbi(cell, getCell(i - 1, o - 1), mt, TB_MATCH, base);
      }

      // Insert from (i-1, o)
      if (i > 0) {
        const it = allIns[i - 1];
        cell = emitStepViterbi(cell, getCell(i - 1, o), it, TB_INSERT, base);
      }

      // Delete from (i, o-1)
      if (o > 0) {
        const dt = allDel[o - 1];
        cell = emitStepViterbi(cell, getCell(i, o - 1), dt, TB_DELETE, base);
      }

      for (let s = 0; s < S; s++) dp[base + s] = cell[s];

      // Propagate silent
      const closed = propagateSilentViterbi(cell, base);
      for (let s = 0; s < S; s++) dp[base + s] = closed[s];
    }
  }

  const score = dp[(Li * (Lo + 1) + Lo) * S + S - 1];
  return { dp, tbSrc, tbType, score };
}

/**
 * Trace back through Viterbi DP to recover the best path.
 */
function traceback(tbSrc, tbType, Li, Lo, S) {
  const path = [];
  let i = Li, o = Lo, s = S - 1;

  while (i > 0 || o > 0 || s !== 0) {
    const idx = (i * (Lo + 1) + o) * S + s;
    const type = tbType[idx];
    const src = tbSrc[idx];

    if (type === TB_NONE || src < 0) break;

    if (type === TB_SILENT) {
      // Silent transition: same (i, o), different state
      s = src;
    } else if (type === TB_MATCH) {
      path.push({ state: s, inputToken: i, outputToken: o });
      i--; o--; s = src;
    } else if (type === TB_INSERT) {
      path.push({ state: s, inputToken: i, outputToken: 0 });
      i--; s = src;
    } else if (type === TB_DELETE) {
      path.push({ state: s, inputToken: 0, outputToken: o });
      o--; s = src;
    }
  }

  path.reverse();
  return path;
}

/**
 * 2D Viterbi with traceback.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq
 * @param {Uint32Array|null} outputSeq
 * @returns {Promise<ViterbiResult>}
 */
export async function viterbi2D(machine, inputSeq, outputSeq) {
  const Li = inputSeq ? inputSeq.length : 0;
  const Lo = outputSeq ? outputSeq.length : 0;

  const { dp, tbSrc, tbType, score } = viterbiForwardWithTraceback(machine, inputSeq, outputSeq);
  const path = traceback(tbSrc, tbType, Li, Lo, machine.nStates);

  return { score, path };
}
