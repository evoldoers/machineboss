// Fused Plan7+transducer batch Forward/Viterbi kernel.
//
// Dispatch: dispatchWorkgroups(B) where B = number of sequences.
// Each thread runs the full DP algorithm for one sequence.
//
// The common functions (logsemiring_plus, propagate_td_silent, etc.)
// are prepended by the JS dispatcher from fused-plan7-common.wgsl.

// Buffers
@group(0) @binding(0) var<storage, read> model: array<f32>;
@group(0) @binding(1) var<storage, read> seqs: array<u32>;     // [offset_0..offset_B, tok_0_0, ...]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;  // B log-likelihoods
@group(0) @binding(3) var<storage, read_write> workspace: array<f32>;

// Model data layout offsets (computed from K, N_AA, S_TD, etc.)
// log_m_to_m:      0
// log_m_to_i:      K
// log_m_to_d:      2*K
// log_i_to_m:      3*K
// log_i_to_i:      4*K
// log_d_to_m:      5*K
// log_d_to_d:      6*K
// log_match_emit:  7*K                        (K * N_AA entries)
// log_ins_emit:    7*K + K*N_AA               (K * N_AA entries)
// log_b_entry:     7*K + 2*K*N_AA             (K entries)
// log_null_emit:   8*K + 2*K*N_AA             (N_AA entries)
// flanking_weights:8*K + 2*K*N_AA + N_AA      (8 entries: n_loop, n_to_b, e_to_cx, e_to_jx, c_loop, c_to_t, j_loop, j_to_b)
// td_log_trans:    8*K + 2*K*N_AA + N_AA + 8  (N_IN_TD * N_OUT_TD * S_TD * S_TD entries)
// td_silent:       8*K + 2*K*N_AA + N_AA + 8 + N_IN_TD*N_OUT_TD*S_TD*S_TD  (S_TD * S_TD)
// aa_to_td_in:     8*K + 2*K*N_AA + N_AA + 8 + N_IN_TD*N_OUT_TD*S_TD*S_TD + S_TD*S_TD  (N_AA, bitcast<u32>)

fn off_m_to_m() -> u32 { return 0u; }
fn off_m_to_i() -> u32 { return K; }
fn off_m_to_d() -> u32 { return 2u * K; }
fn off_i_to_m() -> u32 { return 3u * K; }
fn off_i_to_i() -> u32 { return 4u * K; }
fn off_d_to_m() -> u32 { return 5u * K; }
fn off_d_to_d() -> u32 { return 6u * K; }
fn off_match_emit() -> u32 { return 7u * K; }
fn off_ins_emit() -> u32 { return 7u * K + K * N_AA; }
fn off_b_entry() -> u32 { return 7u * K + 2u * K * N_AA; }
fn off_null_emit() -> u32 { return 8u * K + 2u * K * N_AA; }
fn off_flanking() -> u32 { return 8u * K + 2u * K * N_AA + N_AA; }
fn off_td_trans() -> u32 { return 8u * K + 2u * K * N_AA + N_AA + 8u; }
fn off_td_silent() -> u32 { return off_td_trans() + N_IN_TD * N_OUT_TD * S_TD * S_TD; }
fn off_aa_map() -> u32 { return off_td_silent() + S_TD * S_TD; }

// Flanking state indices
const _N: u32 = 0u;
const _NX: u32 = 1u;
const _B: u32 = 2u;
const _E: u32 = 3u;
const _CX: u32 = 4u;
const _C: u32 = 5u;
const _JX: u32 = 6u;
const _J: u32 = 7u;
const N_FLANKING: u32 = 8u;

// Flanking weight indices within the 8-entry block
const FW_N_LOOP: u32 = 0u;
const FW_N_TO_B: u32 = 1u;
const FW_E_TO_CX: u32 = 2u;
const FW_E_TO_JX: u32 = 3u;
const FW_C_LOOP: u32 = 4u;
const FW_C_TO_T: u32 = 5u;
const FW_J_LOOP: u32 = 6u;
const FW_J_TO_B: u32 = 7u;

fn fw(idx: u32) -> f32 { return model[off_flanking() + idx]; }

// Workspace layout per sequence: core_m(K*S_TD) + core_i(K*S_TD) + core_d(K*S_TD) + flanking(8*S_TD)
fn ws_size() -> u32 { return 3u * K * S_TD + N_FLANKING * S_TD; }
fn ws_core_m(b: u32, k: u32, s: u32) -> u32 { return b * ws_size() + k * S_TD + s; }
fn ws_core_i(b: u32, k: u32, s: u32) -> u32 { return b * ws_size() + K * S_TD + k * S_TD + s; }
fn ws_core_d(b: u32, k: u32, s: u32) -> u32 { return b * ws_size() + 2u * K * S_TD + k * S_TD + s; }
fn ws_flanking(b: u32, f: u32, s: u32) -> u32 { return b * ws_size() + 3u * K * S_TD + f * S_TD + s; }

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let b = gid.x;
  let seq_start = seqs[b];
  let seq_end = seqs[b + 1u];
  let Lo = seq_end - seq_start;
  let B_plus_1 = arrayLength(&output);  // number of sequences (B is output length)

  // Initialize workspace to NEG_INF
  for (var k = 0u; k < K; k++) {
    for (var s = 0u; s < S_TD; s++) {
      workspace[ws_core_m(b, k, s)] = NEG_INF;
      workspace[ws_core_i(b, k, s)] = NEG_INF;
      workspace[ws_core_d(b, k, s)] = NEG_INF;
    }
  }
  for (var f = 0u; f < N_FLANKING; f++) {
    for (var s = 0u; s < S_TD; s++) {
      workspace[ws_flanking(b, f, s)] = NEG_INF;
    }
  }

  // === 8-step flanking initialization ===

  // Step 1: S -> NX (weight 1.0 = log 0.0)
  workspace[ws_flanking(b, _NX, 0u)] = 0.0f;
  // Propagate td_silent at NX
  var v_tmp: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { v_tmp[s] = workspace[ws_flanking(b, _NX, s)]; }
  propagate_td_silent(&v_tmp, off_td_silent(), &model);
  for (var s = 0u; s < S_TD; s++) { workspace[ws_flanking(b, _NX, s)] = v_tmp[s]; }

  // Step 2: NX -> B, NX -> N
  for (var s = 0u; s < S_TD; s++) {
    workspace[ws_flanking(b, _B, s)] = workspace[ws_flanking(b, _NX, s)] + fw(FW_N_TO_B);
    workspace[ws_flanking(b, _N, s)] = workspace[ws_flanking(b, _NX, s)] + fw(FW_N_LOOP);
  }
  // Propagate td_silent at B and N
  for (var s = 0u; s < S_TD; s++) { v_tmp[s] = workspace[ws_flanking(b, _B, s)]; }
  propagate_td_silent(&v_tmp, off_td_silent(), &model);
  for (var s = 0u; s < S_TD; s++) { workspace[ws_flanking(b, _B, s)] = v_tmp[s]; }
  for (var s = 0u; s < S_TD; s++) { v_tmp[s] = workspace[ws_flanking(b, _N, s)]; }
  propagate_td_silent(&v_tmp, off_td_silent(), &model);
  for (var s = 0u; s < S_TD; s++) { workspace[ws_flanking(b, _N, s)] = v_tmp[s]; }

  // Step 3: B -> M_k (pre-emission entry)
  for (var k = 0u; k < K; k++) {
    for (var s = 0u; s < S_TD; s++) {
      workspace[ws_core_m(b, k, s)] = model[off_b_entry() + k] + workspace[ws_flanking(b, _B, s)];
    }
  }

  // Step 4: M_k -> E (pre-emission local exit)
  for (var s = 0u; s < S_TD; s++) { v_tmp[s] = NEG_INF; }
  for (var k = 0u; k < K; k++) {
    var mk: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { mk[s] = workspace[ws_core_m(b, k, s)]; }
    propagate_td_silent(&mk, off_td_silent(), &model);
    for (var s = 0u; s < S_TD; s++) {
      v_tmp[s] = logsemiring_plus(v_tmp[s], mk[s]);
    }
  }
  for (var s = 0u; s < S_TD; s++) { workspace[ws_flanking(b, _E, s)] = v_tmp[s]; }
  for (var s = 0u; s < S_TD; s++) { v_tmp[s] = workspace[ws_flanking(b, _E, s)]; }
  propagate_td_silent(&v_tmp, off_td_silent(), &model);
  for (var s = 0u; s < S_TD; s++) { workspace[ws_flanking(b, _E, s)] = v_tmp[s]; }

  // Step 5: E -> CX, E -> JX
  for (var s = 0u; s < S_TD; s++) {
    workspace[ws_flanking(b, _CX, s)] = workspace[ws_flanking(b, _E, s)] + fw(FW_E_TO_CX);
    workspace[ws_flanking(b, _JX, s)] = workspace[ws_flanking(b, _E, s)] + fw(FW_E_TO_JX);
  }

  // Step 6: JX -> B (multi-hit)
  for (var s = 0u; s < S_TD; s++) {
    let jx_to_b = workspace[ws_flanking(b, _JX, s)] + fw(FW_J_TO_B);
    workspace[ws_flanking(b, _B, s)] = logsemiring_plus(workspace[ws_flanking(b, _B, s)], jx_to_b);
  }

  // Step 7: CX -> C, JX -> J
  for (var s = 0u; s < S_TD; s++) {
    workspace[ws_flanking(b, _C, s)] = workspace[ws_flanking(b, _CX, s)] + fw(FW_C_LOOP);
    workspace[ws_flanking(b, _J, s)] = workspace[ws_flanking(b, _JX, s)] + fw(FW_J_LOOP);
  }

  // Propagate td_silent at CX, JX, C, J
  for (var f_idx = 0u; f_idx < 4u; f_idx++) {
    var fi: u32;
    if (f_idx == 0u) { fi = _CX; }
    else if (f_idx == 1u) { fi = _JX; }
    else if (f_idx == 2u) { fi = _C; }
    else { fi = _J; }
    for (var s = 0u; s < S_TD; s++) { v_tmp[s] = workspace[ws_flanking(b, fi, s)]; }
    propagate_td_silent(&v_tmp, off_td_silent(), &model);
    for (var s = 0u; s < S_TD; s++) { workspace[ws_flanking(b, fi, s)] = v_tmp[s]; }
  }

  // Step 8: Multi-hit B -> M_k entries from JX -> B
  var jx_b: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) {
    jx_b[s] = workspace[ws_flanking(b, _JX, s)] + fw(FW_J_TO_B);
  }
  propagate_td_silent(&jx_b, off_td_silent(), &model);
  for (var k = 0u; k < K; k++) {
    for (var s = 0u; s < S_TD; s++) {
      workspace[ws_core_m(b, k, s)] = logsemiring_plus(
        workspace[ws_core_m(b, k, s)],
        model[off_b_entry() + k] + jx_b[s]
      );
    }
  }

  // Handle initial silent emissions
  emit_silent_core(b);

  // Check empty sequence
  if (Lo == 0u) {
    output[b] = get_terminal_val(b);
    return;
  }

  // === Main DP loop over output positions ===
  for (var p = 0u; p < Lo; p++) {
    let out_tok = seqs[seq_start + p];
    emit_output_step(b, out_tok);
    route_post_emission(b);
    emit_silent_core(b);
  }

  output[b] = get_terminal_val(b);
}

// Plan7 core emits, transducer consumes silently (adds to existing core_m, core_i).
fn emit_silent_core(b: u32) {
  for (var k = 0u; k < K; k++) {
    // Match emit silent
    var v_m: array<f32, 8>;
    var v_result: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { v_m[s] = workspace[ws_core_m(b, k, s)]; }
    td_emit_silent(&v_m, off_match_emit() + k * N_AA, N_AA, &v_result, &model, off_aa_map(), off_td_trans());
    for (var s = 0u; s < S_TD; s++) {
      workspace[ws_core_m(b, k, s)] = logsemiring_plus(workspace[ws_core_m(b, k, s)], v_result[s]);
    }

    // Insert emit silent
    var v_i: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { v_i[s] = workspace[ws_core_i(b, k, s)]; }
    td_emit_silent(&v_i, off_ins_emit() + k * N_AA, N_AA, &v_result, &model, off_aa_map(), off_td_trans());
    for (var s = 0u; s < S_TD; s++) {
      workspace[ws_core_i(b, k, s)] = logsemiring_plus(workspace[ws_core_i(b, k, s)], v_result[s]);
    }
  }
}

// Process one output token: emissions -> transducer produces output.
fn emit_output_step(b: u32, out_tok: u32) {
  // We need temporary storage. Use workspace offset beyond this sequence's area.
  // Actually, use local variables since we process one at a time.

  // Read old state
  // For each core M_k and I_k: emit amino acid, transducer produces output
  for (var k = 0u; k < K; k++) {
    var v_m: array<f32, 8>;
    var new_m: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { v_m[s] = workspace[ws_core_m(b, k, s)]; }
    td_emit_produce(&v_m, off_match_emit() + k * N_AA, N_AA, out_tok, &new_m, &model, off_aa_map(), off_td_trans());

    // Also add transducer-delete path (td produces output without Plan7 emission)
    let td_del_base = off_td_trans() + (0u * N_OUT_TD + out_tok) * S_TD * S_TD;
    for (var dst = 0u; dst < S_TD; dst++) {
      var acc = NEG_INF;
      for (var src = 0u; src < S_TD; src++) {
        acc = logsemiring_plus(acc, v_m[src] + model[td_del_base + src * S_TD + dst]);
      }
      new_m[dst] = logsemiring_plus(new_m[dst], acc);
    }
    for (var s = 0u; s < S_TD; s++) { workspace[ws_core_m(b, k, s)] = new_m[s]; }

    // Insert states
    var v_i: array<f32, 8>;
    var new_i: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { v_i[s] = workspace[ws_core_i(b, k, s)]; }
    td_emit_produce(&v_i, off_ins_emit() + k * N_AA, N_AA, out_tok, &new_i, &model, off_aa_map(), off_td_trans());
    for (var dst = 0u; dst < S_TD; dst++) {
      var acc = NEG_INF;
      for (var src = 0u; src < S_TD; src++) {
        acc = logsemiring_plus(acc, v_i[src] + model[td_del_base + src * S_TD + dst]);
      }
      new_i[dst] = logsemiring_plus(new_i[dst], acc);
    }
    for (var s = 0u; s < S_TD; s++) { workspace[ws_core_i(b, k, s)] = new_i[s]; }

    // Delete states: only td_delete path (D doesn't emit amino acids)
    var v_d: array<f32, 8>;
    var new_d: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { v_d[s] = workspace[ws_core_d(b, k, s)]; }
    for (var dst = 0u; dst < S_TD; dst++) {
      var acc = NEG_INF;
      for (var src = 0u; src < S_TD; src++) {
        acc = logsemiring_plus(acc, v_d[src] + model[td_del_base + src * S_TD + dst]);
      }
      new_d[dst] = acc;
    }
    for (var s = 0u; s < S_TD; s++) { workspace[ws_core_d(b, k, s)] = new_d[s]; }
  }

  // Flanking N/C/J emit background, transducer produces output
  var v_n: array<f32, 8>;
  var n_emitted: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { v_n[s] = workspace[ws_flanking(b, _N, s)]; }
  td_emit_produce(&v_n, off_null_emit(), N_AA, out_tok, &n_emitted, &model, off_aa_map(), off_td_trans());

  var v_c: array<f32, 8>;
  var c_emitted: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { v_c[s] = workspace[ws_flanking(b, _C, s)]; }
  td_emit_produce(&v_c, off_null_emit(), N_AA, out_tok, &c_emitted, &model, off_aa_map(), off_td_trans());

  var v_j: array<f32, 8>;
  var j_emitted: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { v_j[s] = workspace[ws_flanking(b, _J, s)]; }
  td_emit_produce(&v_j, off_null_emit(), N_AA, out_tok, &j_emitted, &model, off_aa_map(), off_td_trans());

  // Transducer-delete for all flanking states
  let td_del_base = off_td_trans() + (0u * N_OUT_TD + out_tok) * S_TD * S_TD;
  for (var f = 0u; f < N_FLANKING; f++) {
    var v_f: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { v_f[s] = workspace[ws_flanking(b, f, s)]; }

    var new_f: array<f32, 8>;
    for (var dst = 0u; dst < S_TD; dst++) {
      var acc = NEG_INF;
      for (var src = 0u; src < S_TD; src++) {
        acc = logsemiring_plus(acc, v_f[src] + model[td_del_base + src * S_TD + dst]);
      }
      // For N, C, J: combine emitted + td_delete
      if (f == _N) {
        new_f[dst] = logsemiring_plus(n_emitted[dst], acc);
      } else if (f == _C) {
        new_f[dst] = logsemiring_plus(c_emitted[dst], acc);
      } else if (f == _J) {
        new_f[dst] = logsemiring_plus(j_emitted[dst], acc);
      } else {
        new_f[dst] = acc;
      }
    }
    for (var s = 0u; s < S_TD; s++) { workspace[ws_flanking(b, f, s)] = new_f[s]; }
  }
}

// Route after emission: inner scan over core, flanking propagation.
fn route_post_emission(b: u32) {
  var m_incoming: array<f32, 8>;
  var d_incoming: array<f32, 8>;
  var e_accum: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) {
    m_incoming[s] = NEG_INF;
    d_incoming[s] = NEG_INF;
    e_accum[s] = NEG_INF;
  }

  // Temporary arrays for arriving pre-emission M values per node
  // We need to store K * S_TD values. Use a second pass approach.
  // First pass: compute routing, track m_arriving per node in workspace core_d
  // (we'll rebuild core_d anyway). Actually, let's store m_arriving temporarily.
  // Since K can be up to 256, we can't use local arrays that large in WGSL.
  // Instead, we'll do two passes or use workspace.

  // Pass 1: compute routing and write m_arriving and new_core_i to workspace
  // We'll use the core_d area temporarily for m_arriving (overwrite is fine since
  // core_d gets rebuilt in the routing scan).

  for (var k = 0u; k < K; k++) {
    // D_k: combine current + incoming from left
    var d_at_k: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) {
      d_at_k[s] = logsemiring_plus(workspace[ws_core_d(b, k, s)], d_incoming[s]);
    }
    propagate_td_silent(&d_at_k, off_td_silent(), &model);

    // Pre-emission M_k from routing
    var m_pre_k: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { m_pre_k[s] = m_incoming[s]; }
    propagate_td_silent(&m_pre_k, off_td_silent(), &model);

    // Post-emission Mx_k
    var mx_k: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { mx_k[s] = workspace[ws_core_m(b, k, s)]; }
    propagate_td_silent(&mx_k, off_td_silent(), &model);

    // Post-emission Ix_k
    var ix_k: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { ix_k[s] = workspace[ws_core_i(b, k, s)]; }
    propagate_td_silent(&ix_k, off_td_silent(), &model);

    // E contributions: pre-emission M_k and D_k
    for (var s = 0u; s < S_TD; s++) {
      e_accum[s] = logsemiring_plus(e_accum[s], m_pre_k[s]);
      e_accum[s] = logsemiring_plus(e_accum[s], d_at_k[s]);
    }

    // Save m_arriving[k] in core_d area (will be overwritten later)
    for (var s = 0u; s < S_TD; s++) {
      workspace[ws_core_d(b, k, s)] = m_pre_k[s];
    }

    // New I_k
    for (var s = 0u; s < S_TD; s++) {
      let i_from_m = mx_k[s] + model[off_m_to_i() + k];
      let i_self = ix_k[s] + model[off_i_to_i() + k];
      workspace[ws_core_i(b, k, s)] = logsemiring_plus(i_from_m, i_self);
    }

    // Compute outgoing routing
    for (var s = 0u; s < S_TD; s++) {
      let m_to_next = mx_k[s] + model[off_m_to_m() + k];
      let m_from_i = ix_k[s] + model[off_i_to_m() + k];
      let m_from_d = d_at_k[s] + model[off_d_to_m() + k];
      let d_from_m = mx_k[s] + model[off_m_to_d() + k];
      let d_from_d = d_at_k[s] + model[off_d_to_d() + k];

      m_incoming[s] = logsemiring_plus(m_to_next, logsemiring_plus(m_from_i, m_from_d));
      d_incoming[s] = logsemiring_plus(d_from_m, d_from_d);
    }
  }

  // Build new flanking
  var new_flanking: array<f32, 64>;  // 8 * S_TD, max S_TD=8
  for (var i = 0u; i < N_FLANKING * S_TD; i++) { new_flanking[i] = NEG_INF; }

  // NX from N, CX from C, JX from J
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_NX * S_TD + s] = workspace[ws_flanking(b, _N, s)];
    new_flanking[_CX * S_TD + s] = workspace[ws_flanking(b, _C, s)];
    new_flanking[_JX * S_TD + s] = workspace[ws_flanking(b, _J, s)];
    new_flanking[_E * S_TD + s] = e_accum[s];
  }

  // Propagate flanking silent (single-pass DAG)
  // E -> CX
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_CX * S_TD + s] = logsemiring_plus(
      new_flanking[_CX * S_TD + s],
      new_flanking[_E * S_TD + s] + fw(FW_E_TO_CX)
    );
  }
  // E -> JX
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_JX * S_TD + s] = logsemiring_plus(
      new_flanking[_JX * S_TD + s],
      new_flanking[_E * S_TD + s] + fw(FW_E_TO_JX)
    );
  }
  // NX -> B
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_B * S_TD + s] = logsemiring_plus(
      new_flanking[_B * S_TD + s],
      new_flanking[_NX * S_TD + s] + fw(FW_N_TO_B)
    );
  }
  // JX -> B
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_B * S_TD + s] = logsemiring_plus(
      new_flanking[_B * S_TD + s],
      new_flanking[_JX * S_TD + s] + fw(FW_J_TO_B)
    );
  }
  // NX -> N
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_N * S_TD + s] = logsemiring_plus(
      new_flanking[_N * S_TD + s],
      new_flanking[_NX * S_TD + s] + fw(FW_N_LOOP)
    );
  }
  // CX -> C
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_C * S_TD + s] = logsemiring_plus(
      new_flanking[_C * S_TD + s],
      new_flanking[_CX * S_TD + s] + fw(FW_C_LOOP)
    );
  }
  // JX -> J
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_J * S_TD + s] = logsemiring_plus(
      new_flanking[_J * S_TD + s],
      new_flanking[_JX * S_TD + s] + fw(FW_J_LOOP)
    );
  }

  // Propagate td_silent at each flanking state
  for (var f = 0u; f < N_FLANKING; f++) {
    var v_f: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { v_f[s] = new_flanking[f * S_TD + s]; }
    propagate_td_silent(&v_f, off_td_silent(), &model);
    for (var s = 0u; s < S_TD; s++) { new_flanking[f * S_TD + s] = v_f[s]; }
  }

  // Write flanking back
  for (var f = 0u; f < N_FLANKING; f++) {
    for (var s = 0u; s < S_TD; s++) {
      workspace[ws_flanking(b, f, s)] = new_flanking[f * S_TD + s];
    }
  }

  // Final core_m = pre-emission from routing + new B entry
  // m_arriving[k] was stored in core_d area
  for (var k = 0u; k < K; k++) {
    for (var s = 0u; s < S_TD; s++) {
      let m_arr = workspace[ws_core_d(b, k, s)]; // m_arriving saved here
      workspace[ws_core_m(b, k, s)] = logsemiring_plus(
        m_arr,
        model[off_b_entry() + k] + new_flanking[_B * S_TD + s]
      );
    }
  }

  // B -> M_k -> E -> CX chain completion
  var e_from_b: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { e_from_b[s] = NEG_INF; }
  for (var k = 0u; k < K; k++) {
    var b_mk: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) {
      b_mk[s] = model[off_b_entry() + k] + new_flanking[_B * S_TD + s];
    }
    propagate_td_silent(&b_mk, off_td_silent(), &model);
    for (var s = 0u; s < S_TD; s++) {
      e_from_b[s] = logsemiring_plus(e_from_b[s], b_mk[s]);
    }
  }
  // Update E
  for (var s = 0u; s < S_TD; s++) {
    workspace[ws_flanking(b, _E, s)] = logsemiring_plus(workspace[ws_flanking(b, _E, s)], e_from_b[s]);
  }
  // E -> CX -> C
  var e_closed: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { e_closed[s] = e_from_b[s]; }
  propagate_td_silent(&e_closed, off_td_silent(), &model);
  for (var s = 0u; s < S_TD; s++) {
    let cx_inc = e_closed[s] + fw(FW_E_TO_CX);
    workspace[ws_flanking(b, _CX, s)] = logsemiring_plus(workspace[ws_flanking(b, _CX, s)], cx_inc);
    let c_inc = cx_inc + fw(FW_C_LOOP);
    workspace[ws_flanking(b, _C, s)] = logsemiring_plus(workspace[ws_flanking(b, _C, s)], c_inc);
  }
  // E -> JX -> J and JX -> B -> M_k (multi-hit)
  for (var s = 0u; s < S_TD; s++) {
    let jx_inc = e_closed[s] + fw(FW_E_TO_JX);
    workspace[ws_flanking(b, _JX, s)] = logsemiring_plus(workspace[ws_flanking(b, _JX, s)], jx_inc);
    let j_inc = jx_inc + fw(FW_J_LOOP);
    workspace[ws_flanking(b, _J, s)] = logsemiring_plus(workspace[ws_flanking(b, _J, s)], j_inc);
    let b_inc = jx_inc + fw(FW_J_TO_B);
    workspace[ws_flanking(b, _B, s)] = logsemiring_plus(workspace[ws_flanking(b, _B, s)], b_inc);
    for (var k = 0u; k < K; k++) {
      workspace[ws_core_m(b, k, s)] = logsemiring_plus(
        workspace[ws_core_m(b, k, s)],
        model[off_b_entry() + k] + b_inc
      );
    }
  }

  // Clear core_d (it was used as temp storage for m_arriving)
  for (var k = 0u; k < K; k++) {
    for (var s = 0u; s < S_TD; s++) {
      workspace[ws_core_d(b, k, s)] = NEG_INF;
    }
  }
}

// Get terminal value: CX -> T
fn get_terminal_val(b: u32) -> f32 {
  var cx_val: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) {
    cx_val[s] = workspace[ws_flanking(b, _CX, s)] + fw(FW_C_TO_T);
  }
  propagate_td_silent(&cx_val, off_td_silent(), &model);
  return cx_val[S_TD - 1u];
}
