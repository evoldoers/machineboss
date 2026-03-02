/**
 * 2D Backward algorithm (CPU fallback, Float64Array).
 *
 * Fills backward DP grid from terminal cell (Li, Lo) to initial cell (0, 0).
 * Uses reverse anti-diagonal order.
 */

import { NEG_INF, makeSemiring } from '../internal/logmath.mjs';
import { propagateSilentBackward, emitStepBackward } from './silent.mjs';

/**
 * Precompute per-position emission-weighted transition matrices.
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

/**
 * 2D Backward algorithm returning full grid.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq
 * @param {Uint32Array|null} outputSeq
 * @param {import('../internal/logmath.mjs').SemiringType} [semiringType='logsumexp']
 * @returns {Promise<{logLikelihood: number, bp: Float64Array}>}
 */
export async function backward2D(machine, inputSeq, outputSeq, semiringType = 'logsumexp') {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans } = machine;
  const semiring = makeSemiring(semiringType);
  const { plus, reduce } = semiring;

  const Li = inputSeq ? inputSeq.length : 0;
  const Lo = outputSeq ? outputSeq.length : 0;

  const silent = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) silent[i] = logTrans[i];

  const bpSize = (Li + 1) * (Lo + 1) * S;
  const bp = new Float64Array(bpSize).fill(NEG_INF);

  function getCell(i, o) {
    const base = (i * (Lo + 1) + o) * S;
    return bp.subarray(base, base + S);
  }

  function setCell(i, o, values) {
    const base = (i * (Lo + 1) + o) * S;
    for (let s = 0; s < S; s++) bp[base + s] = values[s];
  }

  // Initialize terminal cell (Li, Lo): end state S-1 has weight 0
  const termBase = (Li * (Lo + 1) + Lo) * S;
  bp[termBase + S - 1] = 0.0;
  const termCell = propagateSilentBackward(getCell(Li, Lo), silent, S, plus, reduce);
  setCell(Li, Lo, termCell);

  if (Li + Lo === 0) {
    return { logLikelihood: bp[0], bp };
  }

  // Precompute transitions
  const insSlice = Li > 0 ? extractTransSlice(logTrans, nIn, nOut, S, true) : null;
  const delSlice = Lo > 0 ? extractTransSlice(logTrans, nIn, nOut, S, false) : null;
  const allIns = Li > 0 ? precomputeEmitTrans(inputSeq, insSlice, nIn, S, reduce) : null;
  const allDel = Lo > 0 ? precomputeEmitTrans(outputSeq, delSlice, nOut, S, reduce) : null;
  const allMatch = (Li > 0 && Lo > 0) ? precomputeMatch(inputSeq, outputSeq, logTrans, nIn, nOut, S, reduce) : null;

  // Fill in reverse diagonal order
  for (let d = Li + Lo - 1; d >= 0; d--) {
    const iMin = Math.max(0, d - Lo);
    const iMax = Math.min(Li, d);
    for (let i = iMin; i <= iMax; i++) {
      const o = d - i;
      if (o < 0 || o > Lo) continue;
      if (i === Li && o === Lo) continue;

      let cell = new Float64Array(S).fill(NEG_INF);

      // Match to (i+1, o+1)
      if (i < Li && o < Lo) {
        const mt = allMatch[i * Lo + o];
        cell = emitStepBackward(cell, getCell(i + 1, o + 1), mt, S, plus, reduce);
      }

      // Insert to (i+1, o)
      if (i < Li) {
        const it = allIns[i];
        cell = emitStepBackward(cell, getCell(i + 1, o), it, S, plus, reduce);
      }

      // Delete to (i, o+1)
      if (o < Lo) {
        const dt = allDel[o];
        cell = emitStepBackward(cell, getCell(i, o + 1), dt, S, plus, reduce);
      }

      cell = propagateSilentBackward(cell, silent, S, plus, reduce);
      setCell(i, o, cell);
    }
  }

  const ll = bp[0];  // backward probability at state 0, position (0,0)
  return { logLikelihood: ll, bp };
}
