/**
 * Log-space arithmetic utilities.
 *
 * All DP values are stored as log-probabilities. NEG_INF represents
 * log(0) = -Infinity. logaddexp(a,b) = log(exp(a)+exp(b)) is the
 * fundamental "plus" operation in the log-semiring.
 */

export const NEG_INF = -Infinity;

/**
 * log(exp(a) + exp(b)), numerically stable.
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */
export function logaddexp(a, b) {
  if (a === NEG_INF) return b;
  if (b === NEG_INF) return a;
  const m = a > b ? a : b;
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

/**
 * max(a, b) — the "plus" in the max-plus (Viterbi) semiring.
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */
export function logmax(a, b) {
  return a > b ? a : b;
}

/**
 * Semiring abstraction for log-space DP.
 * @typedef {'logsumexp'|'maxplus'} SemiringType
 */

/**
 * @param {SemiringType} type
 * @returns {{ plus: function, reduce: function }}
 */
export function makeSemiring(type) {
  if (type === 'maxplus') {
    return {
      plus: logmax,
      reduce(arr) {
        let m = NEG_INF;
        for (let i = 0; i < arr.length; i++) {
          if (arr[i] > m) m = arr[i];
        }
        return m;
      },
      argreduce(arr) {
        let m = NEG_INF;
        let idx = 0;
        for (let i = 0; i < arr.length; i++) {
          if (arr[i] > m) { m = arr[i]; idx = i; }
        }
        return { value: m, index: idx };
      }
    };
  }
  // logsumexp (default)
  return {
    plus: logaddexp,
    reduce(arr) {
      let m = NEG_INF;
      for (let i = 0; i < arr.length; i++) {
        if (arr[i] > m) m = arr[i];
      }
      if (m === NEG_INF) return NEG_INF;
      let s = 0;
      for (let i = 0; i < arr.length; i++) {
        if (arr[i] !== NEG_INF) s += Math.exp(arr[i] - m);
      }
      return m + Math.log(s);
    },
    argreduce: null  // not meaningful for logsumexp
  };
}

/**
 * Log-semiring matrix multiply: C[i][k] = plus_j(A[i][j] + B[j][k])
 * All matrices are flat Float64Array of size S*S, row-major.
 * @param {Float64Array} A - (S, S) row-major
 * @param {Float64Array} B - (S, S) row-major
 * @param {number} S - state count
 * @param {function} reduce - semiring reduce over array
 * @returns {Float64Array} C - (S, S) row-major
 */
export function matMul(A, B, S, reduce) {
  const C = new Float64Array(S * S);
  const tmp = new Float64Array(S);
  for (let i = 0; i < S; i++) {
    for (let k = 0; k < S; k++) {
      for (let j = 0; j < S; j++) {
        tmp[j] = A[i * S + j] + B[j * S + k];
      }
      C[i * S + k] = reduce(tmp);
    }
  }
  return C;
}

/**
 * Log-space identity matrix (0 on diagonal, NEG_INF elsewhere).
 * @param {number} S
 * @returns {Float64Array}
 */
export function logIdentity(S) {
  const I = new Float64Array(S * S).fill(NEG_INF);
  for (let i = 0; i < S; i++) I[i * S + i] = 0;
  return I;
}

/**
 * Matrix-vector multiply: out[i] = reduce_j(M[i][j] + v[j])
 * @param {Float64Array} M - (S, S) row-major
 * @param {Float64Array} v - (S,)
 * @param {number} S
 * @param {function} reduce
 * @returns {Float64Array}
 */
export function matVecMul(M, v, S, reduce) {
  const out = new Float64Array(S);
  const tmp = new Float64Array(S);
  for (let i = 0; i < S; i++) {
    for (let j = 0; j < S; j++) {
      tmp[j] = M[i * S + j] + v[j];
    }
    out[i] = reduce(tmp);
  }
  return out;
}

/**
 * Transposed matrix-vector multiply: out[j] = reduce_i(M[i][j] + v[i])
 * Used in forward direction: out[dst] = reduce_src(prev[src] + trans[src][dst])
 * @param {Float64Array} M - (S, S) row-major, M[src][dst]
 * @param {Float64Array} v - (S,) source vector
 * @param {number} S
 * @param {function} reduce
 * @returns {Float64Array}
 */
export function vecMatMul(v, M, S, reduce) {
  const out = new Float64Array(S);
  const tmp = new Float64Array(S);
  for (let dst = 0; dst < S; dst++) {
    for (let src = 0; src < S; src++) {
      tmp[src] = v[src] + M[src * S + dst];
    }
    out[dst] = reduce(tmp);
  }
  return out;
}
