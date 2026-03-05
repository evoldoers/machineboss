/**
 * Fused Plan7 Forward/Viterbi kernel (CPU, Float64Array).
 *
 * Direct port of python/machineboss/jax/fused_plan7.py to pure JS.
 * Exploits the linear chain structure of Plan7 profile HMMs for efficient
 * fused DP with a transducer, without materializing the composed state space.
 *
 * Architecture:
 *   - Outer loop over output sequence positions (sequential)
 *   - Inner loop over core profile nodes k=0..K-1
 *   - Flanking states (N, C, J) handled separately at each step
 */

import { NEG_INF, makeSemiring } from '../internal/logmath.mjs';
import { parseHmmer, calcMatchOccupancy } from '../internal/hmmer-parse.mjs';

// Flanking state indices
const _N = 0;
const _NX = 1;
const _B = 2;
const _E = 3;
const _CX = 4;
const _C = 5;
const _JX = 6;
const _J = 7;
const N_FLANKING = 8;

/**
 * Build the fused Plan7 data structure from parsed HMMER model + prepared transducer.
 *
 * @param {{ alph: string[], nodes: Array, null_emit: number[] }} hmmerModel
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} preparedTransducer
 * @param {{ multihit?: boolean, L?: number }} [opts={}]
 * @returns {Object} FusedPlan7Data
 */
export function buildFusedPlan7(hmmerModel, preparedTransducer, opts = {}) {
  const { multihit = false, L = 400 } = opts;
  const K = hmmerModel.nodes.length;
  const n_aa = hmmerModel.alph.length;
  const S_td = preparedTransducer.nStates;
  const n_in_td = preparedTransducer.nInputTokens;
  const n_out_td = preparedTransducer.nOutputTokens;

  const safeLog = x => x > 0 ? Math.log(x) : NEG_INF;

  // Per-node core transitions (log-space)
  const log_m_to_m = new Float64Array(K);
  const log_m_to_i = new Float64Array(K);
  const log_m_to_d = new Float64Array(K);
  const log_i_to_m = new Float64Array(K);
  const log_i_to_i = new Float64Array(K);
  const log_d_to_m = new Float64Array(K);
  const log_d_to_d = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    const n = hmmerModel.nodes[k];
    log_m_to_m[k] = safeLog(n.m_to_m);
    log_m_to_i[k] = safeLog(n.m_to_i);
    log_m_to_d[k] = safeLog(n.m_to_d);
    log_i_to_m[k] = safeLog(n.i_to_m);
    log_i_to_i[k] = safeLog(n.i_to_i);
    log_d_to_m[k] = safeLog(n.d_to_m);
    log_d_to_d[k] = safeLog(n.d_to_d);
  }

  // Per-node emissions (log-space), flat (K * n_aa)
  const log_match_emit = new Float64Array(K * n_aa);
  const log_ins_emit = new Float64Array(K * n_aa);
  for (let k = 0; k < K; k++) {
    for (let a = 0; a < n_aa; a++) {
      log_match_emit[k * n_aa + a] = safeLog(hmmerModel.nodes[k].match_emit[a]);
      log_ins_emit[k * n_aa + a] = safeLog(hmmerModel.nodes[k].ins_emit[a]);
    }
  }

  // Begin transitions (local mode: occupancy-weighted entry)
  const occ = calcMatchOccupancy(hmmerModel);
  let Z = 0;
  for (let k = 1; k < K; k++) {
    Z += occ[k] * (K - k + 1);
  }
  const log_b_entry = new Float64Array(K);
  if (Z > 0) {
    for (let k = 0; k < K - 1; k++) {
      // k in fused array maps to profile node k+1
      log_b_entry[k] = occ[k + 1] > 0 ? Math.log(occ[k + 1] / Z) : NEG_INF;
    }
    log_b_entry[K - 1] = NEG_INF; // last node: no entry in local
  } else {
    log_b_entry.fill(NEG_INF);
  }

  // Flanking weights
  const log_n_loop = Math.log(L / (L + 1));
  const log_n_to_b = Math.log(1.0 / (L + 1));
  const log_c_loop = Math.log(L / (L + 1));
  const log_c_to_t = Math.log(1.0 / (L + 1));

  let log_e_to_cx, log_e_to_jx, log_j_loop, log_j_to_b;
  if (multihit) {
    log_e_to_cx = Math.log(0.5);
    log_e_to_jx = Math.log(0.5);
    log_j_loop = Math.log(L / (L + 1));
    log_j_to_b = Math.log(1.0 / (L + 1));
  } else {
    log_e_to_cx = 0.0; // log(1.0)
    log_e_to_jx = NEG_INF;
    log_j_loop = NEG_INF;
    log_j_to_b = NEG_INF;
  }

  // Null model emissions
  const log_null_emit = new Float64Array(n_aa);
  for (let a = 0; a < n_aa; a++) {
    log_null_emit[a] = safeLog(hmmerModel.null_emit[a]);
  }

  // Map amino acid index -> transducer input token
  const tdInMap = {};
  for (let i = 0; i < preparedTransducer.inputAlphabet.length; i++) {
    tdInMap[preparedTransducer.inputAlphabet[i]] = i;
  }
  const aa_to_td_in = new Uint32Array(n_aa);
  for (let a = 0; a < n_aa; a++) {
    const sym = hmmerModel.alph[a];
    if (sym in tdInMap) {
      aa_to_td_in[a] = tdInMap[sym];
    }
    // else stays 0 (epsilon)
  }

  // Transducer transition tensor: use directly from preparedTransducer.logTrans
  // Layout: logTrans[in_tok * n_out * S * S + out_tok * S * S + src * S + dst]
  const td_log_trans = preparedTransducer.logTrans;

  // Silent transitions: td_log_trans[0, 0, :, :] = first S*S entries
  const td_silent = new Float64Array(S_td * S_td);
  for (let i = 0; i < S_td * S_td; i++) {
    td_silent[i] = td_log_trans[i];
  }

  return {
    K, n_aa, S_td, n_in_td, n_out_td,
    log_m_to_m, log_m_to_i, log_m_to_d,
    log_i_to_m, log_i_to_i,
    log_d_to_m, log_d_to_d,
    log_match_emit, log_ins_emit,
    log_b_entry,
    log_n_loop, log_n_to_b,
    log_e_to_cx, log_e_to_jx,
    log_c_loop, log_c_to_t,
    log_j_loop, log_j_to_b,
    log_null_emit,
    td_log_trans, td_silent,
    aa_to_td_in,
  };
}

// =========================================================================
// Internal helpers
// =========================================================================

/**
 * Transducer matvec: result[dst] = reduce_src(v[src] + trans[src * S + dst])
 */
function tdMatvec(trans, v, S_td, reduce) {
  const result = new Float64Array(S_td);
  const tmp = new Float64Array(S_td);
  for (let dst = 0; dst < S_td; dst++) {
    for (let src = 0; src < S_td; src++) {
      tmp[src] = v[src] + trans[src * S_td + dst];
    }
    result[dst] = reduce(tmp);
  }
  return result;
}

/**
 * Emit amino acid via Plan7 state, transducer consumes and produces output.
 *
 * For each amino acid a: weight = aa_emit_log[a] + td_trans[aa_to_td_in[a], out_tok] @ v_td
 * Then reduce over amino acids.
 */
function tdEmitProduce(v_td, aaEmitLog, aaToTdIn, tdLogTrans, outTok, S_td, n_aa, n_out_td, plus, reduce) {
  const result = new Float64Array(S_td).fill(NEG_INF);
  const tmp = new Float64Array(S_td);
  for (let a = 0; a < n_aa; a++) {
    if (aaEmitLog[a] === NEG_INF) continue;
    // Get transition matrix for this aa's input token and output token
    const inTok = aaToTdIn[a];
    const base = (inTok * n_out_td + outTok) * S_td * S_td;
    // matvec: new_dst = reduce_src(v_td[src] + tdLogTrans[base + src * S_td + dst])
    for (let dst = 0; dst < S_td; dst++) {
      for (let src = 0; src < S_td; src++) {
        tmp[src] = v_td[src] + tdLogTrans[base + src * S_td + dst];
      }
      const tdResult = reduce(tmp);
      result[dst] = plus(result[dst], aaEmitLog[a] + tdResult);
    }
  }
  return result;
}

/**
 * Emit amino acid via Plan7, transducer consumes but produces no output (out_tok=0).
 */
function tdEmitSilent(v_td, aaEmitLog, aaToTdIn, tdLogTrans, S_td, n_aa, n_out_td, plus, reduce) {
  return tdEmitProduce(v_td, aaEmitLog, aaToTdIn, tdLogTrans, 0, S_td, n_aa, n_out_td, plus, reduce);
}

/**
 * Propagate transducer silent transitions to fixed point.
 */
function propagateTdSilent(v_td, tdSilent, S_td, plus, reduce, maxIter = 100) {
  let current = new Float64Array(v_td);
  for (let iter = 0; iter < maxIter; iter++) {
    const update = tdMatvec(tdSilent, current, S_td, reduce);
    const next = new Float64Array(S_td);
    let changed = false;
    for (let i = 0; i < S_td; i++) {
      next[i] = plus(v_td[i], update[i]);
      if (Math.abs(next[i] - current[i]) > 1e-10) changed = true;
    }
    current = next;
    if (!changed) break;
  }
  return current;
}

/**
 * Propagate silent flanking transitions (single-pass DAG).
 *
 * The flanking silent graph is acyclic:
 *   NX -> B, E -> CX, E -> JX, JX -> B
 * so a single pass in topological order suffices.
 * After applying flanking edges, propagate transducer silent at each state.
 */
function propagateFlankingSilent(flanking, fm, tdSilent, S_td, plus, reduce) {
  const result = new Float64Array(flanking);

  // E -> CX (log_e_to_cx)
  for (let s = 0; s < S_td; s++) {
    const v = result[_E * S_td + s] + fm.log_e_to_cx;
    result[_CX * S_td + s] = plus(result[_CX * S_td + s], v);
  }

  // E -> JX (log_e_to_jx)
  for (let s = 0; s < S_td; s++) {
    const v = result[_E * S_td + s] + fm.log_e_to_jx;
    result[_JX * S_td + s] = plus(result[_JX * S_td + s], v);
  }

  // NX -> B (log_n_to_b)
  for (let s = 0; s < S_td; s++) {
    const v = result[_NX * S_td + s] + fm.log_n_to_b;
    result[_B * S_td + s] = plus(result[_B * S_td + s], v);
  }

  // JX -> B (log_j_to_b) — must come after E -> JX
  for (let s = 0; s < S_td; s++) {
    const v = result[_JX * S_td + s] + fm.log_j_to_b;
    result[_B * S_td + s] = plus(result[_B * S_td + s], v);
  }

  // NX -> N (n_loop): sets up N for next emission
  for (let s = 0; s < S_td; s++) {
    const v = result[_NX * S_td + s] + fm.log_n_loop;
    result[_N * S_td + s] = plus(result[_N * S_td + s], v);
  }

  // CX -> C (c_loop): sets up C for next emission — must come after E -> CX
  for (let s = 0; s < S_td; s++) {
    const v = result[_CX * S_td + s] + fm.log_c_loop;
    result[_C * S_td + s] = plus(result[_C * S_td + s], v);
  }

  // JX -> J (j_loop): sets up J for next emission — must come after E -> JX
  for (let s = 0; s < S_td; s++) {
    const v = result[_JX * S_td + s] + fm.log_j_loop;
    result[_J * S_td + s] = plus(result[_J * S_td + s], v);
  }

  // Propagate transducer silent at each flanking state
  for (let f = 0; f < N_FLANKING; f++) {
    const v = result.subarray(f * S_td, (f + 1) * S_td);
    const closed = propagateTdSilent(v, tdSilent, S_td, plus, reduce);
    for (let s = 0; s < S_td; s++) {
      result[f * S_td + s] = closed[s];
    }
  }

  return result;
}

// =========================================================================
// Main DP
// =========================================================================

/**
 * Fused Plan7+transducer Forward algorithm.
 *
 * @param {Object} fm - FusedPlan7Data from buildFusedPlan7
 * @param {Uint32Array} outputSeq - 1-based output token indices
 * @param {string} [semiringType='logsumexp'] - 'logsumexp' or 'maxplus'
 * @returns {number} log-likelihood
 */
export function fusedPlan7Forward(fm, outputSeq, semiringType = 'logsumexp') {
  const { K, n_aa, S_td, n_out_td,
    log_m_to_m, log_m_to_i, log_m_to_d,
    log_i_to_m, log_i_to_i,
    log_d_to_m, log_d_to_d,
    log_match_emit, log_ins_emit,
    log_b_entry,
    log_null_emit,
    td_log_trans, td_silent,
    aa_to_td_in } = fm;

  const semiring = makeSemiring(semiringType);
  const { plus, reduce } = semiring;
  const Lo = outputSeq.length;

  // State representation (flat Float64Arrays):
  //   core_m: K * S_td — M_k pre-emission (ready to emit)
  //   core_i: K * S_td — I_k pre-emission
  //   core_d: K * S_td — D_k (silent)
  //   flanking: N_FLANKING * S_td

  // --- Initialize ---
  // Manual DAG propagation for init (avoids double-counting).
  // All core_m values are PRE-emission (from B -> M_k entry).
  let core_m = new Float64Array(K * S_td).fill(NEG_INF);
  let core_i = new Float64Array(K * S_td).fill(NEG_INF);
  let core_d = new Float64Array(K * S_td).fill(NEG_INF);
  let flanking = new Float64Array(N_FLANKING * S_td).fill(NEG_INF);

  // Step 1: S -> NX; propagate td_silent at NX
  flanking[_NX * S_td + 0] = 0.0;
  {
    const v = new Float64Array(flanking.subarray(_NX * S_td, (_NX + 1) * S_td));
    const closed = propagateTdSilent(v, td_silent, S_td, plus, reduce);
    for (let s = 0; s < S_td; s++) flanking[_NX * S_td + s] = closed[s];
  }

  // Step 2: NX -> B (n_to_b), NX -> N (n_loop)
  for (let s = 0; s < S_td; s++) {
    flanking[_B * S_td + s] = flanking[_NX * S_td + s] + fm.log_n_to_b;
    flanking[_N * S_td + s] = flanking[_NX * S_td + s] + fm.log_n_loop;
  }
  // Propagate td_silent at B and N
  for (const f of [_B, _N]) {
    const v = new Float64Array(flanking.subarray(f * S_td, (f + 1) * S_td));
    const closed = propagateTdSilent(v, td_silent, S_td, plus, reduce);
    for (let s = 0; s < S_td; s++) flanking[f * S_td + s] = closed[s];
  }

  // Step 3: B -> M_k (pre-emission entry)
  for (let k = 0; k < K; k++) {
    for (let s = 0; s < S_td; s++) {
      core_m[k * S_td + s] = log_b_entry[k] + flanking[_B * S_td + s];
    }
  }

  // Step 4: M_k -> E (pre-emission local exit, weight 1)
  let e_val = new Float64Array(S_td).fill(NEG_INF);
  for (let k = 0; k < K; k++) {
    const m_k = propagateTdSilent(
      new Float64Array(core_m.subarray(k * S_td, (k + 1) * S_td)),
      td_silent, S_td, plus, reduce);
    for (let s = 0; s < S_td; s++) {
      e_val[s] = plus(e_val[s], m_k[s]);
    }
  }
  for (let s = 0; s < S_td; s++) flanking[_E * S_td + s] = e_val[s];
  {
    const v = new Float64Array(flanking.subarray(_E * S_td, (_E + 1) * S_td));
    const closed = propagateTdSilent(v, td_silent, S_td, plus, reduce);
    for (let s = 0; s < S_td; s++) flanking[_E * S_td + s] = closed[s];
  }

  // Step 5: E -> CX (e_to_cx), E -> JX (e_to_jx)
  for (let s = 0; s < S_td; s++) {
    flanking[_CX * S_td + s] = flanking[_E * S_td + s] + fm.log_e_to_cx;
    flanking[_JX * S_td + s] = flanking[_E * S_td + s] + fm.log_e_to_jx;
  }

  // Step 6: JX -> B (j_to_b) — adds to existing B for multi-hit
  for (let s = 0; s < S_td; s++) {
    const jx_to_b = flanking[_JX * S_td + s] + fm.log_j_to_b;
    flanking[_B * S_td + s] = plus(flanking[_B * S_td + s], jx_to_b);
  }

  // Step 7: CX -> C (c_loop), JX -> J (j_loop)
  for (let s = 0; s < S_td; s++) {
    flanking[_C * S_td + s] = flanking[_CX * S_td + s] + fm.log_c_loop;
    flanking[_J * S_td + s] = flanking[_JX * S_td + s] + fm.log_j_loop;
  }

  // Propagate td_silent at CX, JX, C, J
  for (const f of [_CX, _JX, _C, _J]) {
    const v = new Float64Array(flanking.subarray(f * S_td, (f + 1) * S_td));
    const closed = propagateTdSilent(v, td_silent, S_td, plus, reduce);
    for (let s = 0; s < S_td; s++) flanking[f * S_td + s] = closed[s];
  }

  // Step 8: Multi-hit B -> M_k entries (from JX -> B, only the increment)
  {
    const jx_to_b = new Float64Array(S_td);
    for (let s = 0; s < S_td; s++) {
      jx_to_b[s] = flanking[_JX * S_td + s] + fm.log_j_to_b;
    }
    const b_inc = propagateTdSilent(jx_to_b, td_silent, S_td, plus, reduce);
    for (let k = 0; k < K; k++) {
      for (let s = 0; s < S_td; s++) {
        core_m[k * S_td + s] = plus(core_m[k * S_td + s], log_b_entry[k] + b_inc[s]);
      }
    }
  }

  // Handle initial silent emissions (Plan7 emits, transducer consumes silently)
  _emitSilentCore(core_m, core_i, fm, plus, reduce);

  if (Lo === 0) {
    return _getTerminalVal(flanking, fm, plus, reduce);
  }

  // Outer loop over output positions
  for (let p = 0; p < Lo; p++) {
    const outTok = outputSeq[p];

    // Emit output
    const emitResult = _emitOutputStep(core_m, core_i, core_d, flanking, outTok, fm, plus, reduce);
    core_m = emitResult.core_m;
    core_i = emitResult.core_i;
    core_d = emitResult.core_d;
    flanking = emitResult.flanking;

    // Route and propagate
    const routeRes = _routePostEmission(core_m, core_i, core_d, flanking, fm, plus, reduce);
    core_m = routeRes.core_m;
    core_i = routeRes.core_i;
    core_d = routeRes.core_d;
    flanking = routeRes.flanking;

    // Silent Plan7 emissions
    _emitSilentCore(core_m, core_i, fm, plus, reduce);
  }

  return _getTerminalVal(flanking, fm, plus, reduce);
}

/**
 * Fused Plan7+transducer Viterbi algorithm.
 *
 * @param {Object} fm - FusedPlan7Data from buildFusedPlan7
 * @param {Uint32Array} outputSeq - 1-based output token indices
 * @returns {number} log Viterbi score
 */
export function fusedPlan7Viterbi(fm, outputSeq) {
  return fusedPlan7Forward(fm, outputSeq, 'maxplus');
}

// =========================================================================
// Internal DP steps
// =========================================================================

/**
 * Plan7 core emits, transducer consumes silently (no output).
 * Modifies core_m and core_i in place.
 */
function _emitSilentCore(core_m, core_i, fm, plus, reduce) {
  const { K, n_aa, S_td, n_out_td, log_match_emit, log_ins_emit,
    aa_to_td_in, td_log_trans } = fm;

  for (let k = 0; k < K; k++) {
    const m_val = core_m.subarray(k * S_td, (k + 1) * S_td);
    const m_emit = log_match_emit.subarray(k * n_aa, (k + 1) * n_aa);
    const m_result = tdEmitSilent(m_val, m_emit, aa_to_td_in, td_log_trans, S_td, n_aa, n_out_td, plus, reduce);
    for (let s = 0; s < S_td; s++) {
      core_m[k * S_td + s] = plus(core_m[k * S_td + s], m_result[s]);
    }

    const i_val = core_i.subarray(k * S_td, (k + 1) * S_td);
    const i_emit = log_ins_emit.subarray(k * n_aa, (k + 1) * n_aa);
    const i_result = tdEmitSilent(i_val, i_emit, aa_to_td_in, td_log_trans, S_td, n_aa, n_out_td, plus, reduce);
    for (let s = 0; s < S_td; s++) {
      core_i[k * S_td + s] = plus(core_i[k * S_td + s], i_result[s]);
    }
  }
}

/**
 * Process one output token: Plan7 emits -> transducer produces output.
 */
function _emitOutputStep(core_m, core_i, core_d, flanking, outTok, fm, plus, reduce) {
  const { K, n_aa, S_td, n_out_td, log_match_emit, log_ins_emit,
    log_null_emit, aa_to_td_in, td_log_trans } = fm;

  const new_core_m = new Float64Array(K * S_td).fill(NEG_INF);
  const new_core_i = new Float64Array(K * S_td).fill(NEG_INF);
  const new_core_d = new Float64Array(K * S_td).fill(NEG_INF);
  const new_flanking = new Float64Array(N_FLANKING * S_td).fill(NEG_INF);

  // 1. Core M_k emits amino acid, transducer produces output
  for (let k = 0; k < K; k++) {
    const m_val = core_m.subarray(k * S_td, (k + 1) * S_td);
    const m_emit = log_match_emit.subarray(k * n_aa, (k + 1) * n_aa);
    const m_emitted = tdEmitProduce(m_val, m_emit, aa_to_td_in, td_log_trans, outTok, S_td, n_aa, n_out_td, plus, reduce);
    for (let s = 0; s < S_td; s++) {
      new_core_m[k * S_td + s] = m_emitted[s];
    }
  }

  // 2. Core I_k emits amino acid, transducer produces output
  for (let k = 0; k < K; k++) {
    const i_val = core_i.subarray(k * S_td, (k + 1) * S_td);
    const i_emit = log_ins_emit.subarray(k * n_aa, (k + 1) * n_aa);
    const i_emitted = tdEmitProduce(i_val, i_emit, aa_to_td_in, td_log_trans, outTok, S_td, n_aa, n_out_td, plus, reduce);
    for (let s = 0; s < S_td; s++) {
      new_core_i[k * S_td + s] = i_emitted[s];
    }
  }

  // 3. Flanking N/C/J emit background, transducer produces output
  const n_val = flanking.subarray(_N * S_td, (_N + 1) * S_td);
  const n_emitted = tdEmitProduce(n_val, log_null_emit, aa_to_td_in, td_log_trans, outTok, S_td, n_aa, n_out_td, plus, reduce);

  const c_val = flanking.subarray(_C * S_td, (_C + 1) * S_td);
  const c_emitted = tdEmitProduce(c_val, log_null_emit, aa_to_td_in, td_log_trans, outTok, S_td, n_aa, n_out_td, plus, reduce);

  const j_val = flanking.subarray(_J * S_td, (_J + 1) * S_td);
  const j_emitted = tdEmitProduce(j_val, log_null_emit, aa_to_td_in, td_log_trans, outTok, S_td, n_aa, n_out_td, plus, reduce);

  // 4. Transducer produces output without Plan7 emission (td delete: in=0, out=outTok)
  const tdDelBase = (0 * n_out_td + outTok) * S_td * S_td;
  const tmp = new Float64Array(S_td);

  // Apply td_delete to all core and flanking states
  for (let k = 0; k < K; k++) {
    const base_k = k * S_td;
    for (let dst = 0; dst < S_td; dst++) {
      for (let src = 0; src < S_td; src++) {
        tmp[src] = core_m[base_k + src] + td_log_trans[tdDelBase + src * S_td + dst];
      }
      new_core_m[base_k + dst] = plus(new_core_m[base_k + dst], reduce(tmp));
    }
    for (let dst = 0; dst < S_td; dst++) {
      for (let src = 0; src < S_td; src++) {
        tmp[src] = core_i[base_k + src] + td_log_trans[tdDelBase + src * S_td + dst];
      }
      new_core_i[base_k + dst] = plus(new_core_i[base_k + dst], reduce(tmp));
    }
    for (let dst = 0; dst < S_td; dst++) {
      for (let src = 0; src < S_td; src++) {
        tmp[src] = core_d[base_k + src] + td_log_trans[tdDelBase + src * S_td + dst];
      }
      new_core_d[base_k + dst] = reduce(tmp);
    }
  }

  for (let f = 0; f < N_FLANKING; f++) {
    const fBase = f * S_td;
    for (let dst = 0; dst < S_td; dst++) {
      for (let src = 0; src < S_td; src++) {
        tmp[src] = flanking[fBase + src] + td_log_trans[tdDelBase + src * S_td + dst];
      }
      const tdDel = reduce(tmp);
      if (f === _N) {
        new_flanking[fBase + dst] = plus(n_emitted[dst], tdDel);
      } else if (f === _C) {
        new_flanking[fBase + dst] = plus(c_emitted[dst], tdDel);
      } else if (f === _J) {
        new_flanking[fBase + dst] = plus(j_emitted[dst], tdDel);
      } else {
        new_flanking[fBase + dst] = tdDel;
      }
    }
  }

  return { core_m: new_core_m, core_i: new_core_i, core_d: new_core_d, flanking: new_flanking };
}

/**
 * Route after emissions: inner scan over core positions for Mx->next, D->next chains.
 * Handles insert self-loops and flanking routing.
 *
 * Key invariant: core_m[k] is POST-emission (at Mx_k). The inner scan's
 * m_incoming carries PRE-emission mass arriving at M_k from left routing.
 * Only pre-emission M_k and D_k contribute to E (local exit).
 * Only post-emission Mx_k is routed through m_to_m/i/d.
 * Pre-emission M_k values persist in the returned core_m for the next emission step.
 */
function _routePostEmission(core_m, core_i, core_d, flanking, fm, plus, reduce) {
  const { K, S_td,
    log_m_to_m, log_m_to_i, log_m_to_d,
    log_i_to_m, log_i_to_i,
    log_d_to_m, log_d_to_d,
    log_b_entry, td_silent } = fm;

  // Inner scan: propagate through core left-to-right
  // m_incoming = pre-emission mass arriving at M_k (from Mx_{k-1}, Ix_{k-1}, D_{k-1})
  // d_incoming = mass arriving at D_k (from Mx_{k-1}, D_{k-1})
  let m_incoming = new Float64Array(S_td).fill(NEG_INF);
  let d_incoming = new Float64Array(S_td).fill(NEG_INF);
  let e_accum = new Float64Array(S_td).fill(NEG_INF);

  // Pre-emission M_k values to persist in core_m for next emission
  const m_arriving = new Float64Array(K * S_td).fill(NEG_INF);
  const new_core_i = new Float64Array(K * S_td).fill(NEG_INF);

  for (let k = 0; k < K; k++) {
    // D_k: combine emit-step D_k with incoming from left
    let d_at_k = new Float64Array(S_td);
    for (let s = 0; s < S_td; s++) {
      d_at_k[s] = plus(core_d[k * S_td + s], d_incoming[s]);
    }
    d_at_k = propagateTdSilent(d_at_k, td_silent, S_td, plus, reduce);

    // Pre-emission M_k from routing (ONLY m_incoming, not core_m[k])
    const m_pre_k = propagateTdSilent(new Float64Array(m_incoming), td_silent, S_td, plus, reduce);

    // Post-emission Mx_k (core_m[k])
    const mx_k = propagateTdSilent(
      new Float64Array(core_m.subarray(k * S_td, (k + 1) * S_td)),
      td_silent, S_td, plus, reduce);

    // Post-emission Ix_k (core_i[k])
    const ix_k = propagateTdSilent(
      new Float64Array(core_i.subarray(k * S_td, (k + 1) * S_td)),
      td_silent, S_td, plus, reduce);

    // E contributions: ONLY pre-emission M_k and D_k (not post-emission Mx_k)
    for (let s = 0; s < S_td; s++) {
      e_accum[s] = plus(e_accum[s], m_pre_k[s]);
      e_accum[s] = plus(e_accum[s], d_at_k[s]);
    }

    // Save pre-emission M_k for persistence
    for (let s = 0; s < S_td; s++) {
      m_arriving[k * S_td + s] = m_pre_k[s];
    }

    // Routing from Mx_k (post-emission only) and D_k and Ix_k
    const new_m_incoming = new Float64Array(S_td).fill(NEG_INF);
    const new_d_incoming = new Float64Array(S_td).fill(NEG_INF);

    for (let s = 0; s < S_td; s++) {
      // From Mx_k (post-emission)
      const m_to_next = mx_k[s] + log_m_to_m[k];
      const i_from_m = mx_k[s] + log_m_to_i[k];
      const d_from_m = mx_k[s] + log_m_to_d[k];

      // From Ix_k (post-emission)
      const m_from_i = ix_k[s] + log_i_to_m[k];
      const i_self = ix_k[s] + log_i_to_i[k];

      // From D_k
      const m_from_d = d_at_k[s] + log_d_to_m[k];
      const d_from_d = d_at_k[s] + log_d_to_d[k];

      // Combined outgoing to next position
      new_m_incoming[s] = plus(m_to_next, plus(m_from_i, m_from_d));
      new_d_incoming[s] = plus(d_from_m, d_from_d);

      // I_k gets contribution from Mx_k -> I_k and self-loop
      new_core_i[k * S_td + s] = plus(i_from_m, i_self);
    }

    m_incoming = new_m_incoming;
    d_incoming = new_d_incoming;
  }

  // Build new flanking
  const new_flanking = new Float64Array(N_FLANKING * S_td).fill(NEG_INF);

  // Nx receives from N (post-emission -> Nx)
  for (let s = 0; s < S_td; s++) {
    new_flanking[_NX * S_td + s] = flanking[_N * S_td + s];
  }

  // Cx receives from C (post-emission -> Cx)
  for (let s = 0; s < S_td; s++) {
    new_flanking[_CX * S_td + s] = flanking[_C * S_td + s];
  }

  // Jx receives from J (post-emission -> Jx)
  for (let s = 0; s < S_td; s++) {
    new_flanking[_JX * S_td + s] = flanking[_J * S_td + s];
  }

  // E receives from core
  for (let s = 0; s < S_td; s++) {
    new_flanking[_E * S_td + s] = e_accum[s];
  }

  // Propagate silent flanking
  const closedFlanking = propagateFlankingSilent(new_flanking, fm, td_silent, S_td, plus, reduce);

  // Final core_m = pre-emission from routing + new B entry
  const new_core_m = new Float64Array(K * S_td).fill(NEG_INF);
  const new_core_d = new Float64Array(K * S_td).fill(NEG_INF);

  const resultFlanking = new Float64Array(closedFlanking);

  const b_val_closed = closedFlanking.subarray(_B * S_td, (_B + 1) * S_td);
  for (let k = 0; k < K; k++) {
    for (let s = 0; s < S_td; s++) {
      new_core_m[k * S_td + s] = plus(
        m_arriving[k * S_td + s],
        log_b_entry[k] + b_val_closed[s]
      );
    }
  }

  // New B -> M_k -> E: the B-entry M_k values are pre-emission and can exit to E.
  // Propagate: B_entry(M_k) -> E -> CX -> C (and E -> JX -> J for multi-hit).
  // This completes the silent path NX -> B -> M_k -> E -> CX within this step.
  let e_from_b = new Float64Array(S_td).fill(NEG_INF);
  for (let k = 0; k < K; k++) {
    const b_mk = propagateTdSilent(
      new Float64Array(S_td).fill(0).map((_, s) => log_b_entry[k] + b_val_closed[s]),
      td_silent, S_td, plus, reduce);
    for (let s = 0; s < S_td; s++) {
      e_from_b[s] = plus(e_from_b[s], b_mk[s]);
    }
  }
  // Add B-entry E contribution to flanking
  for (let s = 0; s < S_td; s++) {
    resultFlanking[_E * S_td + s] = plus(resultFlanking[_E * S_td + s], e_from_b[s]);
  }
  // E -> CX
  const e_closed = propagateTdSilent(e_from_b, td_silent, S_td, plus, reduce);
  for (let s = 0; s < S_td; s++) {
    const cx_inc = e_closed[s] + fm.log_e_to_cx;
    resultFlanking[_CX * S_td + s] = plus(resultFlanking[_CX * S_td + s], cx_inc);
    // CX -> C
    const c_inc = cx_inc + fm.log_c_loop;
    resultFlanking[_C * S_td + s] = plus(resultFlanking[_C * S_td + s], c_inc);
  }
  // E -> JX -> J and JX -> B (multi-hit)
  for (let s = 0; s < S_td; s++) {
    const jx_inc = e_closed[s] + fm.log_e_to_jx;
    resultFlanking[_JX * S_td + s] = plus(resultFlanking[_JX * S_td + s], jx_inc);
    const j_inc = jx_inc + fm.log_j_loop;
    resultFlanking[_J * S_td + s] = plus(resultFlanking[_J * S_td + s], j_inc);
    // JX -> B -> M_k (for multi-hit, adds more core_m entries)
    const b_inc = jx_inc + fm.log_j_to_b;
    resultFlanking[_B * S_td + s] = plus(resultFlanking[_B * S_td + s], b_inc);
    // Additional B -> M_k entries from multi-hit loop
    for (let k = 0; k < K; k++) {
      new_core_m[k * S_td + s] = plus(new_core_m[k * S_td + s], log_b_entry[k] + b_inc);
    }
  }

  return { core_m: new_core_m, core_i: new_core_i, core_d: new_core_d, flanking: resultFlanking };
}

/**
 * Get terminal state value: Cx -> T.
 */
function _getTerminalVal(flanking, fm, plus, reduce) {
  const { S_td, td_silent } = fm;
  const cx_val = new Float64Array(S_td);
  for (let s = 0; s < S_td; s++) {
    cx_val[s] = flanking[_CX * S_td + s] + fm.log_c_to_t;
  }
  const closed = propagateTdSilent(cx_val, td_silent, S_td, plus, reduce);
  return closed[S_td - 1];
}
