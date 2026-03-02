// 2D Backward wavefront kernel.
//
// Each invocation computes one cell (i, o) on anti-diagonal d = i + o.
// Backward: looks forward to (i+1, o+1), (i+1, o), (i, o+1).
// Propagates silent transitions backward (into source states).
//
// Dispatched once per diagonal d = Li+Lo-1 down to 0.

override S: u32;
override LI: u32;
override LO: u32;

@group(0) @binding(0) var<storage, read_write> bp: array<f32>;         // (Li+1)*(Lo+1)*S
@group(0) @binding(1) var<storage, read> all_ins: array<f32>;          // (Li, S, S)
@group(0) @binding(2) var<storage, read> all_del: array<f32>;          // (Lo, S, S)
@group(0) @binding(3) var<storage, read> all_match: array<f32>;        // (Li, Lo, S, S)
@group(0) @binding(4) var<storage, read> silent: array<f32>;           // (S, S)
@group(0) @binding(5) var<uniform> diag_params: DiagParams;

struct DiagParams {
  d: u32,
}

const NEG_INF: f32 = -3.402823466e+38f;

fn logaddexp(a: f32, b: f32) -> f32 {
  if (a <= NEG_INF) { return b; }
  if (b <= NEG_INF) { return a; }
  let m = max(a, b);
  return m + log(exp(a - m) + exp(b - m));
}

fn bp_idx(i: u32, o: u32, s: u32) -> u32 {
  return (i * (LO + 1u) + o) * S + s;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let d = diag_params.d;
  let thread_idx = gid.x;

  let i_min = select(0u, d - LO, d > LO);
  let i = i_min + thread_idx;
  let o = d - i;

  if (i > LI || o > LO) { return; }
  if (i == LI && o == LO) { return; }

  let ss = S * S;

  var cell: array<f32, 64>;
  for (var s = 0u; s < S; s++) {
    cell[s] = NEG_INF;
  }

  // Match to (i+1, o+1)
  if (i < LI && o < LO) {
    let match_base = (i * LO + o) * ss;
    for (var src = 0u; src < S; src++) {
      var acc = NEG_INF;
      for (var dst = 0u; dst < S; dst++) {
        let future = bp[bp_idx(i + 1u, o + 1u, dst)];
        let w = all_match[match_base + src * S + dst];
        acc = logaddexp(acc, w + future);
      }
      cell[src] = logaddexp(cell[src], acc);
    }
  }

  // Insert to (i+1, o)
  if (i < LI) {
    let ins_base = i * ss;
    for (var src = 0u; src < S; src++) {
      var acc = NEG_INF;
      for (var dst = 0u; dst < S; dst++) {
        let future = bp[bp_idx(i + 1u, o, dst)];
        let w = all_ins[ins_base + src * S + dst];
        acc = logaddexp(acc, w + future);
      }
      cell[src] = logaddexp(cell[src], acc);
    }
  }

  // Delete to (i, o+1)
  if (o < LO) {
    let del_base = o * ss;
    for (var src = 0u; src < S; src++) {
      var acc = NEG_INF;
      for (var dst = 0u; dst < S; dst++) {
        let future = bp[bp_idx(i, o + 1u, dst)];
        let w = all_del[del_base + src * S + dst];
        acc = logaddexp(acc, w + future);
      }
      cell[src] = logaddexp(cell[src], acc);
    }
  }

  // Propagate silent transitions backward
  for (var iter = 0u; iter < S; iter++) {
    var changed = false;
    for (var src = 0u; src < S; src++) {
      var acc = NEG_INF;
      for (var dst = 0u; dst < S; dst++) {
        let val = silent[src * S + dst] + cell[dst];
        acc = logaddexp(acc, val);
      }
      let new_val = logaddexp(cell[src], acc);
      if (abs(new_val - cell[src]) > 1e-7f) {
        cell[src] = new_val;
        changed = true;
      }
    }
    if (!changed) { break; }
  }

  // Write back
  for (var s = 0u; s < S; s++) {
    bp[bp_idx(i, o, s)] = cell[s];
  }
}
