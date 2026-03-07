// Fused Plan7+transducer common WGSL functions.
// Prepended to batch and single-sequence kernels by the JS dispatcher.

const NEG_INF: f32 = -3.402823466e+38f;

override K: u32;          // number of profile nodes
override S_TD: u32;       // transducer state count
override N_AA: u32;       // amino acid alphabet size
override N_OUT_TD: u32;   // transducer output token count
override N_IN_TD: u32;    // transducer input token count
override USE_MAX: u32 = 0u; // 0 = logsumexp, 1 = max-plus

// Log-semiring plus: logsumexp or max, controlled by USE_MAX override.
fn logsemiring_plus(a: f32, b: f32) -> f32 {
  if (USE_MAX == 1u) {
    return max(a, b);
  }
  if (a <= NEG_INF) { return b; }
  if (b <= NEG_INF) { return a; }
  let m = max(a, b);
  return m + log(exp(a - m) + exp(b - m));
}

// Log-semiring reduce over sources: result[dst] = plus_src(v[src] + trans[src * S_TD + dst])
fn td_matvec_dst(v: ptr<function, array<f32, 8>>, trans_base: u32, dst: u32, model: ptr<storage, array<f32>, read>) -> f32 {
  var acc = NEG_INF;
  for (var src = 0u; src < S_TD; src++) {
    acc = logsemiring_plus(acc, (*v)[src] + (*model)[trans_base + src * S_TD + dst]);
  }
  return acc;
}

// Propagate transducer silent transitions to fixed point.
// v is modified in-place. Uses model data starting at td_silent_offset.
fn propagate_td_silent(v: ptr<function, array<f32, 8>>, td_silent_offset: u32, model: ptr<storage, array<f32>, read>) {
  var base_v: array<f32, 8>;
  for (var s = 0u; s < S_TD; s++) {
    base_v[s] = (*v)[s];
  }
  for (var iter = 0u; iter < S_TD; iter++) {
    var changed = false;
    for (var dst = 0u; dst < S_TD; dst++) {
      var acc = NEG_INF;
      for (var src = 0u; src < S_TD; src++) {
        acc = logsemiring_plus(acc, (*v)[src] + (*model)[td_silent_offset + src * S_TD + dst]);
      }
      let new_val = logsemiring_plus(base_v[dst], acc);
      if (abs(new_val - (*v)[dst]) > 1e-6f) {
        (*v)[dst] = new_val;
        changed = true;
      }
    }
    if (!changed) { break; }
  }
}

// Plan7 emits amino acid, transducer consumes aa and produces output token out_tok.
// Returns result in out_v. aa_emit starts at aa_emit_offset in model (K*N_AA or N_AA entries).
fn td_emit_produce(
  v: ptr<function, array<f32, 8>>,
  aa_emit_offset: u32,
  n_emit: u32,
  out_tok: u32,
  out_v: ptr<function, array<f32, 8>>,
  model: ptr<storage, array<f32>, read>,
  aa_map_offset: u32,
  td_trans_offset: u32,
) {
  for (var s = 0u; s < S_TD; s++) {
    (*out_v)[s] = NEG_INF;
  }
  for (var a = 0u; a < n_emit; a++) {
    let emit_w = (*model)[aa_emit_offset + a];
    if (emit_w <= NEG_INF) { continue; }
    let in_tok = bitcast<u32>((*model)[aa_map_offset + a]);
    let base = td_trans_offset + (in_tok * N_OUT_TD + out_tok) * S_TD * S_TD;
    for (var dst = 0u; dst < S_TD; dst++) {
      var acc = NEG_INF;
      for (var src = 0u; src < S_TD; src++) {
        acc = logsemiring_plus(acc, (*v)[src] + (*model)[base + src * S_TD + dst]);
      }
      (*out_v)[dst] = logsemiring_plus((*out_v)[dst], emit_w + acc);
    }
  }
}

// Plan7 emits amino acid, transducer consumes silently (out_tok = 0).
fn td_emit_silent(
  v: ptr<function, array<f32, 8>>,
  aa_emit_offset: u32,
  n_emit: u32,
  out_v: ptr<function, array<f32, 8>>,
  model: ptr<storage, array<f32>, read>,
  aa_map_offset: u32,
  td_trans_offset: u32,
) {
  td_emit_produce(v, aa_emit_offset, n_emit, 0u, out_v, model, aa_map_offset, td_trans_offset);
}
