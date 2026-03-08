/**
 * Wavefront beam-Viterbi alignment for cyclic transducers (CPU).
 *
 * Organizes DP along anti-diagonal wavefronts d = i + j.
 * Consuming transitions advance the wavefront; silent transitions
 * are resolved within each wavefront via iterative closure.
 *
 * Works on cyclic machines (e.g. TKF92, Plan7) where standard
 * Viterbi requires topological sort.
 */

import { NEG_INF } from '../internal/logmath.mjs';

/**
 * @typedef {Object} BeamAlignResult
 * @property {number} score - Viterbi log-score
 * @property {Array<{inPos: number, outPos: number, srcState: number, dstState: number, inputToken: number, outputToken: number}>} path
 */

/**
 * Wavefront beam-Viterbi alignment.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq - 1-based input tokens
 * @param {Uint32Array|null} outputSeq - 1-based output tokens
 * @param {number} [beamWidth=100] - max cells per wavefront
 * @returns {Promise<BeamAlignResult>}
 */
export async function beamAlign2D(machine, inputSeq, outputSeq, beamWidth = 100) {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans } = machine;
  const Li = inputSeq ? inputSeq.length : 0;
  const Lo = outputSeq ? outputSeq.length : 0;
  const startState = 0;
  const endState = S - 1;
  const maxD = Li + Lo;

  // Build outgoing transition list per state (only finite-weight transitions)
  const outgoing = new Array(S);
  for (let src = 0; src < S; src++) {
    const trans = [];
    for (let inTok = 0; inTok < nIn; inTok++) {
      for (let outTok = 0; outTok < nOut; outTok++) {
        for (let dst = 0; dst < S; dst++) {
          const w = logTrans[((inTok * nOut + outTok) * S + src) * S + dst];
          if (w > NEG_INF / 2) {
            trans.push({ inTok, outTok, dst, w });
          }
        }
      }
    }
    outgoing[src] = trans;
  }

  // Beam cell: [inPos, outPos, state, score, parentWave, parentIdx, srcState]
  // Store as arrays of arrays for flexibility
  class Beam {
    constructor() {
      this.cells = [];
      this.keyMap = new Map(); // "inPos,outPos,state" -> index
    }

    _key(inPos, outPos, state) {
      return inPos * (Lo + 1) * S + outPos * S + state;
    }

    addOrUpdate(inPos, outPos, state, score, parentWave, parentIdx, srcState) {
      const key = this._key(inPos, outPos, state);
      if (this.keyMap.has(key)) {
        const idx = this.keyMap.get(key);
        if (score > this.cells[idx][3]) {
          this.cells[idx] = [inPos, outPos, state, score, parentWave, parentIdx, srcState];
        }
      } else {
        this.keyMap.set(key, this.cells.length);
        this.cells.push([inPos, outPos, state, score, parentWave, parentIdx, srcState]);
      }
    }

    prune(width, waveIdx) {
      if (this.cells.length <= width) return;
      // Sort indices by score descending
      const indices = Array.from({ length: this.cells.length }, (_, i) => i);
      indices.sort((a, b) => this.cells[b][3] - this.cells[a][3]);
      const kept = indices.slice(0, width);
      const reindex = new Map();
      const newCells = [];
      for (let i = 0; i < kept.length; i++) {
        reindex.set(kept[i], i);
        newCells.push(this.cells[kept[i]]);
      }
      // Fix up parent indices: remap same-wavefront parents, leave others as-is
      for (const cell of newCells) {
        if (cell[4] === waveIdx) {
          // Parent is in this wavefront — remap or invalidate
          if (reindex.has(cell[5])) {
            cell[5] = reindex.get(cell[5]);
          } else {
            // Parent was pruned — walk back to find a surviving ancestor
            // or invalidate (path will be incomplete but score is still valid)
            cell[4] = -1;
            cell[5] = -1;
          }
        }
      }
      this.cells = newCells;
      this.keyMap = new Map();
      for (let i = 0; i < this.cells.length; i++) {
        const c = this.cells[i];
        this.keyMap.set(this._key(c[0], c[1], c[2]), i);
      }
    }
  }

  function silentClosure(beam, waveIdx) {
    let changed = true;
    while (changed) {
      changed = false;
      const n = beam.cells.length;
      for (let ci = 0; ci < n; ci++) {
        const cell = beam.cells[ci];
        const srcInPos = cell[0];
        const srcOutPos = cell[1];
        const srcState = cell[2];
        const srcScore = cell[3];
        for (const t of outgoing[srcState]) {
          if (t.inTok === 0 && t.outTok === 0) {
            const newScore = srcScore + t.w;
            const key = beam._key(srcInPos, srcOutPos, t.dst);
            if (beam.keyMap.has(key)) {
              const idx = beam.keyMap.get(key);
              if (newScore > beam.cells[idx][3]) {
                beam.cells[idx] = [srcInPos, srcOutPos, t.dst, newScore, waveIdx, ci, srcState];
                changed = true;
              }
            } else {
              beam.keyMap.set(key, beam.cells.length);
              beam.cells.push([srcInPos, srcOutPos, t.dst, newScore, waveIdx, ci, srcState]);
              changed = true;
            }
          }
        }
      }
    }
  }

  const wavefronts = new Array(maxD + 1);

  // Initialize wavefront 0
  const beam0 = new Beam();
  beam0.addOrUpdate(0, 0, startState, 0.0, -1, -1, 0);
  silentClosure(beam0, 0);
  beam0.prune(beamWidth, 0);
  wavefronts[0] = beam0;

  // Fill wavefronts
  for (let d = 1; d <= maxD; d++) {
    const beam = new Beam();

    // From d-1: insert (input-only) and delete (output-only)
    if (d - 1 >= 0 && wavefronts[d - 1]) {
      const prev = wavefronts[d - 1];
      for (let ci = 0; ci < prev.cells.length; ci++) {
        const cell = prev.cells[ci];
        const srcInPos = cell[0], srcOutPos = cell[1], srcState = cell[2], srcScore = cell[3];
        for (const t of outgoing[srcState]) {
          if (t.inTok !== 0 && t.outTok === 0) {
            // Insert: consumes input only
            const newInPos = srcInPos + 1;
            if (newInPos <= Li && inputSeq[newInPos - 1] === t.inTok) {
              beam.addOrUpdate(newInPos, srcOutPos, t.dst, srcScore + t.w, d - 1, ci, srcState);
            }
          } else if (t.inTok === 0 && t.outTok !== 0) {
            // Delete: consumes output only
            const newOutPos = srcOutPos + 1;
            if (newOutPos <= Lo && outputSeq[newOutPos - 1] === t.outTok) {
              beam.addOrUpdate(srcInPos, newOutPos, t.dst, srcScore + t.w, d - 1, ci, srcState);
            }
          }
        }
      }
    }

    // From d-2: match (both)
    if (d - 2 >= 0 && wavefronts[d - 2]) {
      const prev = wavefronts[d - 2];
      for (let ci = 0; ci < prev.cells.length; ci++) {
        const cell = prev.cells[ci];
        const srcInPos = cell[0], srcOutPos = cell[1], srcState = cell[2], srcScore = cell[3];
        for (const t of outgoing[srcState]) {
          if (t.inTok !== 0 && t.outTok !== 0) {
            const newInPos = srcInPos + 1;
            const newOutPos = srcOutPos + 1;
            if (newInPos <= Li && newOutPos <= Lo
                && inputSeq[newInPos - 1] === t.inTok
                && outputSeq[newOutPos - 1] === t.outTok) {
              beam.addOrUpdate(newInPos, newOutPos, t.dst, srcScore + t.w, d - 2, ci, srcState);
            }
          }
        }
      }
    }

    // Silent closure
    silentClosure(beam, d);
    beam.prune(beamWidth, d);
    wavefronts[d] = beam;
  }

  // Find end cell
  const endBeam = wavefronts[maxD];
  if (!endBeam) return { score: -Infinity, path: [] };

  const endKey = endBeam._key(Li, Lo, endState);
  if (!endBeam.keyMap.has(endKey)) return { score: -Infinity, path: [] };

  const endIdx = endBeam.keyMap.get(endKey);
  const score = endBeam.cells[endIdx][3];

  // Traceback
  const path = [];
  let curWave = maxD;
  let curIdx = endIdx;
  while (curWave >= 0 && curIdx >= 0) {
    const cell = wavefronts[curWave].cells[curIdx];
    const parentWave = cell[4];
    const parentIdx = cell[5];
    const srcState = cell[6];

    if (parentWave < 0) break;

    const parentCell = wavefronts[parentWave].cells[parentIdx];
    const di = cell[0] - parentCell[0];
    const dj = cell[1] - parentCell[1];
    const inTok = di > 0 ? inputSeq[cell[0] - 1] : 0;
    const outTok = dj > 0 ? outputSeq[cell[1] - 1] : 0;

    path.push({
      inPos: cell[0], outPos: cell[1],
      srcState, dstState: cell[2],
      inputToken: inTok, outputToken: outTok
    });

    curWave = parentWave;
    curIdx = parentIdx;
  }

  path.reverse();
  return { score, path };
}
