// Fused Plan7+transducer single-sequence kernel.
//
// Dispatch: dispatchWorkgroups(1) — single workgroup, outer loop over positions.
// Workgroup size: K_PADDED threads (one per profile node, padded to power-of-2).
//
// Each output position proceeds in 3 phases with workgroupBarrier() between them:
//   Phase 1: Emission (parallel over K threads)
//   Phase 2: Routing (sequential, thread 0 only)
//   Phase 3: Flanking propagation and write-back (thread 0)
//
// The common functions (logsemiring_plus, propagate_td_silent, etc.)
// are prepended by the JS dispatcher from fused-plan7-common.wgsl.

override K_PADDED: u32;   // workgroup size (next power of 2 >= K)
override LO: u32;         // output sequence length

// Buffers
@group(0) @binding(0) var<storage, read> model: array<f32>;
@group(0) @binding(1) var<storage, read> seq: array<u32>;          // LO output tokens
@group(0) @binding(2) var<storage, read_write> result: array<f32>; // 1 float output

// Shared workgroup memory for DP state
// core_m, core_i, core_d: K * S_TD each
// flanking: 8 * S_TD
// Temporaries for emission results: K * S_TD each for new_m, new_i, new_d
var<workgroup> core_m: array<f32, 2048>;    // K * S_TD, max 256*8
var<workgroup> core_i: array<f32, 2048>;
var<workgroup> core_d: array<f32, 2048>;
var<workgroup> flanking: array<f32, 64>;    // 8 * S_TD, max 8*8
var<workgroup> new_core_m: array<f32, 2048>;
var<workgroup> new_core_i: array<f32, 2048>;
var<workgroup> new_core_d: array<f32, 2048>;

// Model data layout offsets (same as batch shader)
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

const FW_N_LOOP: u32 = 0u;
const FW_N_TO_B: u32 = 1u;
const FW_E_TO_CX: u32 = 2u;
const FW_E_TO_JX: u32 = 3u;
const FW_C_LOOP: u32 = 4u;
const FW_C_TO_T: u32 = 5u;
const FW_J_LOOP: u32 = 6u;
const FW_J_TO_B: u32 = 7u;

fn fw(idx: u32) -> f32 { return model[off_flanking() + idx]; }

@compute @workgroup_size(K_PADDED)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  let k = lid.x;
  let is_active = k < K;

  // === Initialization (thread 0 handles flanking, all threads handle their node) ===
  if (is_active) {
    for (var s = 0u; s < S_TD; s++) {
      core_m[k * S_TD + s] = NEG_INF;
      core_i[k * S_TD + s] = NEG_INF;
      core_d[k * S_TD + s] = NEG_INF;
    }
  }
  if (k == 0u) {
    for (var f = 0u; f < N_FLANKING; f++) {
      for (var s = 0u; s < S_TD; s++) {
        flanking[f * S_TD + s] = NEG_INF;
      }
    }
    // 8-step init (same as batch, done by thread 0)
    init_flanking();
  }
  workgroupBarrier();

  // Set initial core_m from B entry
  if (is_active) {
    for (var s = 0u; s < S_TD; s++) {
      core_m[k * S_TD + s] = model[off_b_entry() + k] + flanking[_B * S_TD + s];
    }
  }
  workgroupBarrier();

  // Step 4: M_k -> E (thread 0 accumulates)
  if (k == 0u) {
    var e_val: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { e_val[s] = NEG_INF; }
    for (var kk = 0u; kk < K; kk++) {
      var mk: array<f32, 8>;
      for (var s = 0u; s < S_TD; s++) { mk[s] = core_m[kk * S_TD + s]; }
      propagate_td_silent(&mk, off_td_silent(), &model);
      for (var s = 0u; s < S_TD; s++) {
        e_val[s] = logsemiring_plus(e_val[s], mk[s]);
      }
    }
    for (var s = 0u; s < S_TD; s++) { flanking[_E * S_TD + s] = e_val[s]; }
    var v_e: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { v_e[s] = flanking[_E * S_TD + s]; }
    propagate_td_silent(&v_e, off_td_silent(), &model);
    for (var s = 0u; s < S_TD; s++) { flanking[_E * S_TD + s] = v_e[s]; }

    // Steps 5-8
    complete_init_flanking();
  }
  workgroupBarrier();

  // Update core_m from multi-hit B entry (step 8)
  if (is_active) {
    for (var s = 0u; s < S_TD; s++) {
      // flanking[_B] already includes JX->B contribution from complete_init_flanking
      core_m[k * S_TD + s] = logsemiring_plus(
        core_m[k * S_TD + s],
        model[off_b_entry() + k] + flanking[_B * S_TD + s]
      );
      // Subtract original B contribution to avoid double-counting
      // Actually, complete_init_flanking updates B in place, so core_m needs
      // to be set to the full B value. Let's just set it directly.
    }
  }
  workgroupBarrier();

  // Actually we need to be more careful. The init already set core_m = b_entry + B.
  // Then complete_init_flanking may have updated B (from JX->B multi-hit).
  // We need to redo: core_m[k] = b_entry[k] + updated_B.
  // The safest approach: thread 0 signals via flanking, then all threads reset.
  if (is_active) {
    for (var s = 0u; s < S_TD; s++) {
      core_m[k * S_TD + s] = model[off_b_entry() + k] + flanking[_B * S_TD + s];
    }
  }
  workgroupBarrier();

  // Initial silent emissions (parallel: each thread handles its node k)
  if (is_active) {
    emit_silent_node(k);
  }
  workgroupBarrier();

  // Check empty sequence
  if (LO == 0u) {
    if (k == 0u) {
      result[0] = get_terminal_val_shared();
    }
    return;
  }

  // === Main DP loop ===
  for (var p = 0u; p < LO; p++) {
    let out_tok = seq[p];

    // Phase 1: Emission (parallel over K)
    if (is_active) {
      emit_output_node(k, out_tok);
    }
    // Thread 0 handles flanking emission
    if (k == 0u) {
      emit_output_flanking(out_tok);
    }
    workgroupBarrier();

    // Copy new values to main arrays
    if (is_active) {
      for (var s = 0u; s < S_TD; s++) {
        core_m[k * S_TD + s] = new_core_m[k * S_TD + s];
        core_i[k * S_TD + s] = new_core_i[k * S_TD + s];
        core_d[k * S_TD + s] = new_core_d[k * S_TD + s];
      }
    }
    workgroupBarrier();

    // Phase 2: Routing (thread 0, sequential scan)
    if (k == 0u) {
      route_post_emission_shared();
    }
    workgroupBarrier();

    // Phase 3: Silent emissions (parallel)
    if (is_active) {
      emit_silent_node(k);
    }
    workgroupBarrier();
  }

  if (k == 0u) {
    result[0] = get_terminal_val_shared();
  }
}

// Initialize flanking states (steps 1-3, thread 0 only)
fn init_flanking() {
  // Step 1: S -> NX
  flanking[_NX * S_TD + 0u] = 0.0f;
  var v_tmp: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { v_tmp[s] = flanking[_NX * S_TD + s]; }
  propagate_td_silent(&v_tmp, off_td_silent(), &model);
  for (var s = 0u; s < S_TD; s++) { flanking[_NX * S_TD + s] = v_tmp[s]; }

  // Step 2: NX -> B, NX -> N
  for (var s = 0u; s < S_TD; s++) {
    flanking[_B * S_TD + s] = flanking[_NX * S_TD + s] + fw(FW_N_TO_B);
    flanking[_N * S_TD + s] = flanking[_NX * S_TD + s] + fw(FW_N_LOOP);
  }
  for (var s = 0u; s < S_TD; s++) { v_tmp[s] = flanking[_B * S_TD + s]; }
  propagate_td_silent(&v_tmp, off_td_silent(), &model);
  for (var s = 0u; s < S_TD; s++) { flanking[_B * S_TD + s] = v_tmp[s]; }
  for (var s = 0u; s < S_TD; s++) { v_tmp[s] = flanking[_N * S_TD + s]; }
  propagate_td_silent(&v_tmp, off_td_silent(), &model);
  for (var s = 0u; s < S_TD; s++) { flanking[_N * S_TD + s] = v_tmp[s]; }
}

// Complete flanking init (steps 5-8, thread 0 only, after E is computed)
fn complete_init_flanking() {
  // Step 5: E -> CX, E -> JX
  for (var s = 0u; s < S_TD; s++) {
    flanking[_CX * S_TD + s] = flanking[_E * S_TD + s] + fw(FW_E_TO_CX);
    flanking[_JX * S_TD + s] = flanking[_E * S_TD + s] + fw(FW_E_TO_JX);
  }

  // Step 6: JX -> B
  for (var s = 0u; s < S_TD; s++) {
    let jx_to_b = flanking[_JX * S_TD + s] + fw(FW_J_TO_B);
    flanking[_B * S_TD + s] = logsemiring_plus(flanking[_B * S_TD + s], jx_to_b);
  }

  // Step 7: CX -> C, JX -> J
  for (var s = 0u; s < S_TD; s++) {
    flanking[_C * S_TD + s] = flanking[_CX * S_TD + s] + fw(FW_C_LOOP);
    flanking[_J * S_TD + s] = flanking[_JX * S_TD + s] + fw(FW_J_LOOP);
  }

  // Propagate td_silent at CX, JX, C, J
  var v_tmp: array<f32, 8>;
  for (var f_idx = 0u; f_idx < 4u; f_idx++) {
    var fi: u32;
    if (f_idx == 0u) { fi = _CX; }
    else if (f_idx == 1u) { fi = _JX; }
    else if (f_idx == 2u) { fi = _C; }
    else { fi = _J; }
    for (var s = 0u; s < S_TD; s++) { v_tmp[s] = flanking[fi * S_TD + s]; }
    propagate_td_silent(&v_tmp, off_td_silent(), &model);
    for (var s = 0u; s < S_TD; s++) { flanking[fi * S_TD + s] = v_tmp[s]; }
  }
}

// Emit silent for one node (called by each thread for its k)
fn emit_silent_node(k: u32) {
  // Match emit silent
  var v_m: array<f32, 8>;
  var v_result: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { v_m[s] = core_m[k * S_TD + s]; }
  td_emit_silent(&v_m, off_match_emit() + k * N_AA, N_AA, &v_result, &model, off_aa_map(), off_td_trans());
  for (var s = 0u; s < S_TD; s++) {
    core_m[k * S_TD + s] = logsemiring_plus(core_m[k * S_TD + s], v_result[s]);
  }

  // Insert emit silent
  var v_i: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { v_i[s] = core_i[k * S_TD + s]; }
  td_emit_silent(&v_i, off_ins_emit() + k * N_AA, N_AA, &v_result, &model, off_aa_map(), off_td_trans());
  for (var s = 0u; s < S_TD; s++) {
    core_i[k * S_TD + s] = logsemiring_plus(core_i[k * S_TD + s], v_result[s]);
  }
}

// Phase 1: Emission for one node k (writes to new_core_m/i/d)
fn emit_output_node(k: u32, out_tok: u32) {
  var v_m: array<f32, 8>;
  var emit_m: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { v_m[s] = core_m[k * S_TD + s]; }
  td_emit_produce(&v_m, off_match_emit() + k * N_AA, N_AA, out_tok, &emit_m, &model, off_aa_map(), off_td_trans());

  // Transducer-delete path
  let td_del_base = off_td_trans() + (0u * N_OUT_TD + out_tok) * S_TD * S_TD;
  for (var dst = 0u; dst < S_TD; dst++) {
    var acc = NEG_INF;
    for (var src = 0u; src < S_TD; src++) {
      acc = logsemiring_plus(acc, v_m[src] + model[td_del_base + src * S_TD + dst]);
    }
    emit_m[dst] = logsemiring_plus(emit_m[dst], acc);
  }
  for (var s = 0u; s < S_TD; s++) { new_core_m[k * S_TD + s] = emit_m[s]; }

  // Insert
  var v_i: array<f32, 8>;
  var emit_i: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { v_i[s] = core_i[k * S_TD + s]; }
  td_emit_produce(&v_i, off_ins_emit() + k * N_AA, N_AA, out_tok, &emit_i, &model, off_aa_map(), off_td_trans());
  for (var dst = 0u; dst < S_TD; dst++) {
    var acc = NEG_INF;
    for (var src = 0u; src < S_TD; src++) {
      acc = logsemiring_plus(acc, v_i[src] + model[td_del_base + src * S_TD + dst]);
    }
    emit_i[dst] = logsemiring_plus(emit_i[dst], acc);
  }
  for (var s = 0u; s < S_TD; s++) { new_core_i[k * S_TD + s] = emit_i[s]; }

  // Delete: only td_delete
  var v_d: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { v_d[s] = core_d[k * S_TD + s]; }
  for (var dst = 0u; dst < S_TD; dst++) {
    var acc = NEG_INF;
    for (var src = 0u; src < S_TD; src++) {
      acc = logsemiring_plus(acc, v_d[src] + model[td_del_base + src * S_TD + dst]);
    }
    new_core_d[k * S_TD + dst] = acc;
  }
}

// Phase 1: Flanking emission (thread 0 only, writes to flanking directly)
fn emit_output_flanking(out_tok: u32) {
  let td_del_base = off_td_trans() + (0u * N_OUT_TD + out_tok) * S_TD * S_TD;

  // N, C, J emit background
  var v_n: array<f32, 8>;
  var n_emitted: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { v_n[s] = flanking[_N * S_TD + s]; }
  td_emit_produce(&v_n, off_null_emit(), N_AA, out_tok, &n_emitted, &model, off_aa_map(), off_td_trans());

  var v_c: array<f32, 8>;
  var c_emitted: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { v_c[s] = flanking[_C * S_TD + s]; }
  td_emit_produce(&v_c, off_null_emit(), N_AA, out_tok, &c_emitted, &model, off_aa_map(), off_td_trans());

  var v_j: array<f32, 8>;
  var j_emitted: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { v_j[s] = flanking[_J * S_TD + s]; }
  td_emit_produce(&v_j, off_null_emit(), N_AA, out_tok, &j_emitted, &model, off_aa_map(), off_td_trans());

  // Transducer-delete for all flanking states
  for (var f = 0u; f < N_FLANKING; f++) {
    var v_f: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { v_f[s] = flanking[f * S_TD + s]; }
    for (var dst = 0u; dst < S_TD; dst++) {
      var acc = NEG_INF;
      for (var src = 0u; src < S_TD; src++) {
        acc = logsemiring_plus(acc, v_f[src] + model[td_del_base + src * S_TD + dst]);
      }
      if (f == _N) {
        flanking[f * S_TD + dst] = logsemiring_plus(n_emitted[dst], acc);
      } else if (f == _C) {
        flanking[f * S_TD + dst] = logsemiring_plus(c_emitted[dst], acc);
      } else if (f == _J) {
        flanking[f * S_TD + dst] = logsemiring_plus(j_emitted[dst], acc);
      } else {
        flanking[f * S_TD + dst] = acc;
      }
    }
  }
}

// Phase 2: Routing (thread 0 only, sequential scan over nodes)
fn route_post_emission_shared() {
  var m_incoming: array<f32, 8>;
  var d_incoming: array<f32, 8>;
  var e_accum: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) {
    m_incoming[s] = NEG_INF;
    d_incoming[s] = NEG_INF;
    e_accum[s] = NEG_INF;
  }

  for (var kk = 0u; kk < K; kk++) {
    // D_k
    var d_at_k: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) {
      d_at_k[s] = logsemiring_plus(core_d[kk * S_TD + s], d_incoming[s]);
    }
    propagate_td_silent(&d_at_k, off_td_silent(), &model);

    // Pre-emission M_k
    var m_pre_k: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { m_pre_k[s] = m_incoming[s]; }
    propagate_td_silent(&m_pre_k, off_td_silent(), &model);

    // Post-emission Mx_k
    var mx_k: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { mx_k[s] = core_m[kk * S_TD + s]; }
    propagate_td_silent(&mx_k, off_td_silent(), &model);

    // Post-emission Ix_k
    var ix_k: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) { ix_k[s] = core_i[kk * S_TD + s]; }
    propagate_td_silent(&ix_k, off_td_silent(), &model);

    // E contributions
    for (var s = 0u; s < S_TD; s++) {
      e_accum[s] = logsemiring_plus(e_accum[s], m_pre_k[s]);
      e_accum[s] = logsemiring_plus(e_accum[s], d_at_k[s]);
    }

    // Save m_arriving in core_d (temp)
    for (var s = 0u; s < S_TD; s++) {
      core_d[kk * S_TD + s] = m_pre_k[s];
    }

    // New I_k
    for (var s = 0u; s < S_TD; s++) {
      core_i[kk * S_TD + s] = logsemiring_plus(
        mx_k[s] + model[off_m_to_i() + kk],
        ix_k[s] + model[off_i_to_i() + kk]
      );
    }

    // Routing
    for (var s = 0u; s < S_TD; s++) {
      m_incoming[s] = logsemiring_plus(
        mx_k[s] + model[off_m_to_m() + kk],
        logsemiring_plus(
          ix_k[s] + model[off_i_to_m() + kk],
          d_at_k[s] + model[off_d_to_m() + kk]
        )
      );
      d_incoming[s] = logsemiring_plus(
        mx_k[s] + model[off_m_to_d() + kk],
        d_at_k[s] + model[off_d_to_d() + kk]
      );
    }
  }

  // Build new flanking
  var new_flanking: array<f32, 64>;
  for (var i = 0u; i < N_FLANKING * S_TD; i++) { new_flanking[i] = NEG_INF; }

  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_NX * S_TD + s] = flanking[_N * S_TD + s];
    new_flanking[_CX * S_TD + s] = flanking[_C * S_TD + s];
    new_flanking[_JX * S_TD + s] = flanking[_J * S_TD + s];
    new_flanking[_E * S_TD + s] = e_accum[s];
  }

  // Propagate flanking silent (single-pass DAG)
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_CX * S_TD + s] = logsemiring_plus(new_flanking[_CX * S_TD + s], new_flanking[_E * S_TD + s] + fw(FW_E_TO_CX));
  }
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_JX * S_TD + s] = logsemiring_plus(new_flanking[_JX * S_TD + s], new_flanking[_E * S_TD + s] + fw(FW_E_TO_JX));
  }
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_B * S_TD + s] = logsemiring_plus(new_flanking[_B * S_TD + s], new_flanking[_NX * S_TD + s] + fw(FW_N_TO_B));
  }
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_B * S_TD + s] = logsemiring_plus(new_flanking[_B * S_TD + s], new_flanking[_JX * S_TD + s] + fw(FW_J_TO_B));
  }
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_N * S_TD + s] = logsemiring_plus(new_flanking[_N * S_TD + s], new_flanking[_NX * S_TD + s] + fw(FW_N_LOOP));
  }
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_C * S_TD + s] = logsemiring_plus(new_flanking[_C * S_TD + s], new_flanking[_CX * S_TD + s] + fw(FW_C_LOOP));
  }
  for (var s = 0u; s < S_TD; s++) {
    new_flanking[_J * S_TD + s] = logsemiring_plus(new_flanking[_J * S_TD + s], new_flanking[_JX * S_TD + s] + fw(FW_J_LOOP));
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
      flanking[f * S_TD + s] = new_flanking[f * S_TD + s];
    }
  }

  // Final core_m = m_arriving + B entry
  for (var kk = 0u; kk < K; kk++) {
    for (var s = 0u; s < S_TD; s++) {
      let m_arr = core_d[kk * S_TD + s]; // m_arriving
      core_m[kk * S_TD + s] = logsemiring_plus(
        m_arr,
        model[off_b_entry() + kk] + new_flanking[_B * S_TD + s]
      );
    }
  }

  // B -> M_k -> E -> CX chain
  var e_from_b: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { e_from_b[s] = NEG_INF; }
  for (var kk = 0u; kk < K; kk++) {
    var b_mk: array<f32, 8>;
    for (var s = 0u; s < S_TD; s++) {
      b_mk[s] = model[off_b_entry() + kk] + new_flanking[_B * S_TD + s];
    }
    propagate_td_silent(&b_mk, off_td_silent(), &model);
    for (var s = 0u; s < S_TD; s++) {
      e_from_b[s] = logsemiring_plus(e_from_b[s], b_mk[s]);
    }
  }
  for (var s = 0u; s < S_TD; s++) {
    flanking[_E * S_TD + s] = logsemiring_plus(flanking[_E * S_TD + s], e_from_b[s]);
  }
  var e_closed: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) { e_closed[s] = e_from_b[s]; }
  propagate_td_silent(&e_closed, off_td_silent(), &model);
  for (var s = 0u; s < S_TD; s++) {
    let cx_inc = e_closed[s] + fw(FW_E_TO_CX);
    flanking[_CX * S_TD + s] = logsemiring_plus(flanking[_CX * S_TD + s], cx_inc);
    flanking[_C * S_TD + s] = logsemiring_plus(flanking[_C * S_TD + s], cx_inc + fw(FW_C_LOOP));
  }
  for (var s = 0u; s < S_TD; s++) {
    let jx_inc = e_closed[s] + fw(FW_E_TO_JX);
    flanking[_JX * S_TD + s] = logsemiring_plus(flanking[_JX * S_TD + s], jx_inc);
    flanking[_J * S_TD + s] = logsemiring_plus(flanking[_J * S_TD + s], jx_inc + fw(FW_J_LOOP));
    let b_inc = jx_inc + fw(FW_J_TO_B);
    flanking[_B * S_TD + s] = logsemiring_plus(flanking[_B * S_TD + s], b_inc);
    for (var kk = 0u; kk < K; kk++) {
      core_m[kk * S_TD + s] = logsemiring_plus(
        core_m[kk * S_TD + s],
        model[off_b_entry() + kk] + b_inc
      );
    }
  }

  // Clear core_d
  for (var kk = 0u; kk < K; kk++) {
    for (var s = 0u; s < S_TD; s++) {
      core_d[kk * S_TD + s] = NEG_INF;
    }
  }
}

// Terminal value: CX -> T
fn get_terminal_val_shared() -> f32 {
  var cx_val: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) {
    cx_val[s] = flanking[_CX * S_TD + s] + fw(FW_C_TO_T);
  }
  propagate_td_silent(&cx_val, off_td_silent(), &model);
  return cx_val[S_TD - 1u];
}
