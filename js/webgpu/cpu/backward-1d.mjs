/**
 * 1D Backward algorithm (CPU fallback, Float64Array).
 *
 * For generator (input=null) or recognizer (output=null) machines.
 * Sequential scan from end to start: O(L * S^2).
 */

import { NEG_INF, makeSemiring } from '../internal/logmath.mjs';
import { propagateSilentBackward, emitStepBackward } from './silent.mjs';

/**
 * Precompute per-position emission-weighted transition matrices for 1D.
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
 * 1D Backward algorithm returning full grid.
 *
 * bp[p][s] = backward probability at position p, state s.
 * bp[L] = terminal (end state = S-1).
 * bp[0] should equal the Forward log-likelihood at state 0.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq
 * @param {Uint32Array|null} outputSeq
 * @param {import('../internal/logmath.mjs').SemiringType} [semiringType='logsumexp']
 * @returns {Promise<{logLikelihood: number, bp: Float64Array}>}
 */
export async function backward1D(machine, inputSeq, outputSeq, semiringType = 'logsumexp') {
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

  // bp[p * S + s] for p = 0..L
  const bp = new Float64Array((L + 1) * S).fill(NEG_INF);

  // Initialize terminal position L: end state = S-1
  bp[L * S + S - 1] = 0.0;
  const termCell = propagateSilentBackward(bp.subarray(L * S, (L + 1) * S), silent, S, plus, reduce);
  for (let s = 0; s < S; s++) bp[L * S + s] = termCell[s];

  if (L > 0) {
    const allTrans = precomputeEmitTrans1D(seq, transSlice, nTok, S, reduce);

    for (let p = L - 1; p >= 0; p--) {
      const future = bp.subarray((p + 1) * S, (p + 2) * S);
      let cell = new Float64Array(S).fill(NEG_INF);
      // Backward: cell[src] += trans[src][dst] + future[dst]
      cell = emitStepBackward(cell, future, allTrans[p], S, plus, reduce);
      cell = propagateSilentBackward(cell, silent, S, plus, reduce);
      for (let s = 0; s < S; s++) bp[p * S + s] = cell[s];
    }
  }

  const ll = bp[0];  // backward at position 0, state 0
  return { logLikelihood: ll, bp };
}
