/**
 * Machine preparation: JSON machine + params -> dense log_trans tensor.
 *
 * Converts a Machine Boss JSON transducer into the dense representation
 * needed by DP algorithms: log_trans[in_tok][out_tok][src][dst] as a
 * flat Float32Array (for GPU) or Float64Array (for CPU).
 *
 * Token convention: token 0 is the "null" (epsilon/gap) token.
 * Real tokens are 1-based indices into the sorted alphabet.
 */

import { NEG_INF } from './logmath.mjs';

/**
 * Evaluate a weight expression to a float given parameter values.
 * Mirrors python/machineboss/weight.py:evaluate().
 *
 * @param {*} w - Weight expression (number, string, or object)
 * @param {Object<string,number>} params - Parameter name -> value
 * @param {Object<string,*>} [defs] - Function/parameter definitions from machine
 * @returns {number}
 */
export function evaluateWeight(w, params, defs) {
  if (w === null || w === undefined) return 0;
  if (typeof w === 'number') return w;
  if (typeof w === 'boolean') return w ? 1.0 : 0.0;
  if (typeof w === 'string') {
    if (params != null && w in params) return params[w];
    if (defs != null && w in defs) return evaluateWeight(defs[w], params, defs);
    throw new Error(`Unknown parameter "${w}". Pass it in the params object, e.g. { ${w}: 0.5 }`);
  }
  if (typeof w === 'object') {
    if ('*' in w) {
      const [a, b] = w['*'];
      return evaluateWeight(a, params, defs) * evaluateWeight(b, params, defs);
    }
    if ('+' in w) {
      const [a, b] = w['+'];
      return evaluateWeight(a, params, defs) + evaluateWeight(b, params, defs);
    }
    if ('-' in w) {
      const [a, b] = w['-'];
      return evaluateWeight(a, params, defs) - evaluateWeight(b, params, defs);
    }
    if ('/' in w) {
      const [a, b] = w['/'];
      return evaluateWeight(a, params, defs) / evaluateWeight(b, params, defs);
    }
    if ('pow' in w) {
      const [base, exp] = w['pow'];
      return Math.pow(evaluateWeight(base, params, defs), evaluateWeight(exp, params, defs));
    }
    if ('log' in w) {
      return Math.log(evaluateWeight(w['log'], params, defs));
    }
    if ('exp' in w) {
      return Math.exp(evaluateWeight(w['exp'], params, defs));
    }
    if ('not' in w) {
      return 1.0 - evaluateWeight(w['not'], params, defs);
    }
    throw new Error(`Unsupported weight operator "${Object.keys(w).join(', ')}". Supported: *, +, -, /, pow, log, exp, not`);
  }
  throw new TypeError(`Unsupported weight expression type: ${typeof w}`);
}

/**
 * Prepared machine: dense transition tensor + metadata.
 * @typedef {Object} PreparedMachine
 * @property {number} nStates - Number of states (S)
 * @property {number} nInputTokens - Number of input tokens including null (n_in)
 * @property {number} nOutputTokens - Number of output tokens including null (n_out)
 * @property {string[]} inputAlphabet - Input token symbols (index 0 = null)
 * @property {string[]} outputAlphabet - Output token symbols (index 0 = null)
 * @property {Float64Array} logTrans - Dense tensor (n_in * n_out * S * S), row-major
 * @property {Float32Array} logTransF32 - Same as logTrans but Float32 for GPU
 */

/**
 * Build sorted token alphabet from machine transitions.
 * Token 0 is reserved for null (epsilon). Real tokens are 1-based.
 *
 * @param {Array} states - Machine JSON states array
 * @param {'in'|'out'} direction
 * @returns {string[]} alphabet where index 0 is null string
 */
function buildAlphabet(states, direction) {
  const tokens = new Set();
  for (const state of states) {
    for (const trans of (state.trans || [])) {
      const tok = trans[direction];
      if (tok) tokens.add(tok);
    }
  }
  const sorted = Array.from(tokens).sort();
  return ['', ...sorted];  // index 0 = null/epsilon
}

/**
 * Prepare a machine JSON + params into a dense log transition tensor.
 *
 * @param {Object} machineJSON - Machine Boss JSON object
 * @param {Object<string,number>} [params={}] - Parameter values
 * @returns {PreparedMachine}
 */
export function prepareMachine(machineJSON, params = {}) {
  const states = machineJSON.state;
  const defs = machineJSON.defs || {};
  const S = states.length;

  const inputAlphabet = buildAlphabet(states, 'in');
  const outputAlphabet = buildAlphabet(states, 'out');
  const nIn = inputAlphabet.length;
  const nOut = outputAlphabet.length;

  // Build token -> index maps
  const inTokIdx = {};
  for (let i = 0; i < nIn; i++) inTokIdx[inputAlphabet[i]] = i;
  const outTokIdx = {};
  for (let i = 0; i < nOut; i++) outTokIdx[outputAlphabet[i]] = i;

  // Resolve state name references to indices
  const nameToIdx = {};
  for (let i = 0; i < S; i++) {
    const name = states[i].id;
    if (name !== undefined) {
      const key = Array.isArray(name) ? JSON.stringify(name) : name;
      nameToIdx[key] = i;
    }
    nameToIdx[i] = i;
  }

  function resolveDest(dest) {
    if (typeof dest === 'number') return dest;
    const key = Array.isArray(dest) ? JSON.stringify(dest) : dest;
    if (key in nameToIdx) return nameToIdx[key];
    throw new Error(`Transition references unknown state "${dest}". Check that the "to" field matches a state "id" in your machine JSON.`);
  }

  // Allocate dense tensor: logTrans[inTok * nOut * S * S + outTok * S * S + src * S + dst]
  const size = nIn * nOut * S * S;
  const logTrans = new Float64Array(size).fill(NEG_INF);

  // Fill from transitions
  for (let src = 0; src < S; src++) {
    const trans = states[src].trans || [];
    for (const t of trans) {
      const dst = resolveDest(t.to);
      const inIdx = t.in ? inTokIdx[t.in] : 0;
      const outIdx = t.out ? outTokIdx[t.out] : 0;
      const logWeight = Math.log(evaluateWeight(t.weight != null ? t.weight : 1, params, defs));
      const idx = ((inIdx * nOut + outIdx) * S + src) * S + dst;
      // logaddexp in case there are duplicate transitions
      if (logTrans[idx] === NEG_INF) {
        logTrans[idx] = logWeight;
      } else {
        const a = logTrans[idx], b = logWeight;
        const m = a > b ? a : b;
        logTrans[idx] = m + Math.log(Math.exp(a - m) + Math.exp(b - m));
      }
    }
  }

  // Float32 version for GPU
  const logTransF32 = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    logTransF32[i] = logTrans[i] === NEG_INF ? -Infinity : logTrans[i];
  }

  return {
    nStates: S,
    nInputTokens: nIn,
    nOutputTokens: nOut,
    inputAlphabet,
    outputAlphabet,
    logTrans,
    logTransF32,
  };
}

/**
 * Look up token index for a symbol in an alphabet.
 * @param {string[]} alphabet
 * @param {string} symbol
 * @returns {number} 1-based token index
 */
export function tokenIndex(alphabet, symbol) {
  const idx = alphabet.indexOf(symbol);
  if (idx < 0) throw new Error(`Unknown symbol "${symbol}". Valid symbols: ${alphabet.slice(1).join(', ')}`);
  return idx;
}

/**
 * Convert a string/array of symbols to Uint32Array of token indices.
 * @param {string|string[]} seq - Sequence of symbols
 * @param {string[]} alphabet - Token alphabet (index 0 = null)
 * @returns {Uint32Array} 1-based token indices
 */
export function tokenize(seq, alphabet) {
  const symbols = typeof seq === 'string' ? seq.split('') : seq;
  const result = new Uint32Array(symbols.length);
  for (let i = 0; i < symbols.length; i++) {
    result[i] = tokenIndex(alphabet, symbols[i]);
  }
  return result;
}

/**
 * Extract the silent transition submatrix logTrans[0][0][src][dst].
 * @param {Float64Array} logTrans
 * @param {number} nOut
 * @param {number} S
 * @returns {Float64Array} S*S matrix
 */
export function extractSilent(logTrans, nOut, S) {
  const silent = new Float64Array(S * S);
  const base = 0; // inTok=0, outTok=0
  for (let i = 0; i < S * S; i++) {
    silent[i] = logTrans[base + i];
  }
  return silent;
}
