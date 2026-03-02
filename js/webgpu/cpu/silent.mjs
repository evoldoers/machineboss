/**
 * Silent transition closure computation and propagation.
 *
 * Silent transitions (in=null, out=null) form a subgraph that must
 * be "closed over" before DP proceeds. The closure matrix S* = I + S + S^2 + ...
 * is the Kleene star: it gives the total weight of all silent paths.
 *
 * For acyclic machines, this terminates in at most S-1 iterations.
 */

import { NEG_INF, logaddexp, matMul, logIdentity } from '../internal/logmath.mjs';

/**
 * Compute the Kleene star (closure) of a silent transition matrix.
 * S* = I + S + S^2 + ... (in log-semiring).
 *
 * @param {Float64Array} silent - (S, S) silent transition log-weights, row-major
 * @param {number} S - number of states
 * @param {function} reduce - semiring reduce function
 * @param {number} [maxIter=100] - maximum iterations
 * @returns {Float64Array} (S, S) closure matrix in log-space
 */
export function silentClosure(silent, S, reduce, maxIter = 100) {
  const identity = logIdentity(S);
  let result = new Float64Array(identity);
  let power = new Float64Array(identity);

  for (let iter = 0; iter < maxIter; iter++) {
    const nextPower = matMul(power, silent, S, reduce);
    const newResult = new Float64Array(S * S);
    let changed = false;
    for (let i = 0; i < S * S; i++) {
      newResult[i] = logaddexp(result[i], nextPower[i]);
      if (Math.abs(newResult[i] - result[i]) > 1e-10) changed = true;
    }
    result = newResult;
    power = nextPower;
    if (!changed) break;
  }
  return result;
}

/**
 * Propagate silent transitions forward within a single DP cell.
 * Accumulates into destination states until convergence.
 *
 * cell'[dst] = plus(cell[dst], reduce_src(cell[src] + silent[src][dst]))
 *
 * @param {Float64Array} cell - (S,) log-probabilities (modified in place)
 * @param {Float64Array} silent - (S, S) silent transition log-weights
 * @param {number} S
 * @param {function} plus - semiring plus (logaddexp or max)
 * @param {function} reduce - semiring reduce
 * @param {number} [maxIter=100]
 * @returns {Float64Array} updated cell
 */
export function propagateSilent(cell, silent, S, plus, reduce, maxIter = 100) {
  let current = new Float64Array(cell);
  const tmp = new Float64Array(S);

  for (let iter = 0; iter < maxIter; iter++) {
    const next = new Float64Array(S);
    let changed = false;
    for (let dst = 0; dst < S; dst++) {
      // incoming[dst] = reduce_src(current[src] + silent[src * S + dst])
      for (let src = 0; src < S; src++) {
        tmp[src] = current[src] + silent[src * S + dst];
      }
      const incoming = reduce(tmp);
      next[dst] = plus(cell[dst], incoming);
      if (Math.abs(next[dst] - current[dst]) > 1e-10) changed = true;
    }
    current = next;
    if (!changed) break;
  }
  return current;
}

/**
 * Propagate silent transitions backward within a single DP cell.
 * Accumulates into source states until convergence.
 *
 * cell'[src] = plus(cell[src], reduce_dst(silent[src][dst] + cell[dst]))
 *
 * @param {Float64Array} cell - (S,) log-probabilities
 * @param {Float64Array} silent - (S, S) silent transition log-weights
 * @param {number} S
 * @param {function} plus
 * @param {function} reduce
 * @param {number} [maxIter=100]
 * @returns {Float64Array} updated cell
 */
export function propagateSilentBackward(cell, silent, S, plus, reduce, maxIter = 100) {
  let current = new Float64Array(cell);
  const tmp = new Float64Array(S);

  for (let iter = 0; iter < maxIter; iter++) {
    const next = new Float64Array(S);
    let changed = false;
    for (let src = 0; src < S; src++) {
      // incoming[src] = reduce_dst(silent[src * S + dst] + current[dst])
      for (let dst = 0; dst < S; dst++) {
        tmp[dst] = silent[src * S + dst] + current[dst];
      }
      const incoming = reduce(tmp);
      next[src] = plus(cell[src], incoming);
      if (Math.abs(next[src] - current[src]) > 1e-10) changed = true;
    }
    current = next;
    if (!changed) break;
  }
  return current;
}

/**
 * Apply emitting transitions forward: out[dst] = reduce_src(prev[src] + trans[src][dst])
 * then combine with cell: result[dst] = plus(cell[dst], out[dst])
 *
 * @param {Float64Array} cell - (S,) current cell being filled
 * @param {Float64Array} prev - (S,) predecessor cell
 * @param {Float64Array} trans - (S, S) transition matrix [src][dst]
 * @param {number} S
 * @param {function} plus
 * @param {function} reduce
 * @returns {Float64Array} updated cell
 */
export function emitStepForward(cell, prev, trans, S, plus, reduce) {
  const result = new Float64Array(S);
  const tmp = new Float64Array(S);
  for (let dst = 0; dst < S; dst++) {
    for (let src = 0; src < S; src++) {
      tmp[src] = prev[src] + trans[src * S + dst];
    }
    const incoming = reduce(tmp);
    result[dst] = plus(cell[dst], incoming);
  }
  return result;
}

/**
 * Apply emitting transitions backward: out[src] = reduce_dst(trans[src][dst] + future[dst])
 * then combine with cell: result[src] = plus(cell[src], out[src])
 *
 * @param {Float64Array} cell - (S,) current cell being filled
 * @param {Float64Array} future - (S,) successor cell
 * @param {Float64Array} trans - (S, S) transition matrix [src][dst]
 * @param {number} S
 * @param {function} plus
 * @param {function} reduce
 * @returns {Float64Array} updated cell
 */
export function emitStepBackward(cell, future, trans, S, plus, reduce) {
  const result = new Float64Array(S);
  const tmp = new Float64Array(S);
  for (let src = 0; src < S; src++) {
    for (let dst = 0; dst < S; dst++) {
      tmp[dst] = trans[src * S + dst] + future[dst];
    }
    const incoming = reduce(tmp);
    result[src] = plus(cell[src], incoming);
  }
  return result;
}
