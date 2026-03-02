// 2D Forward wavefront kernel.
//
// Each invocation computes one cell (i, o) on anti-diagonal d = i + o.
// Combines match from (i-1, o-1), insert from (i-1, o), delete from (i, o-1),
// then propagates silent transitions.
//
// Dispatched once per diagonal d = 1..Li+Lo, with ceil(min(Li,Lo)) workgroups.

override S: u32;
override LI: u32;  // input length
override LO: u32;  // output length

@group(0) @binding(0) var<storage, read_write> dp: array<f32>;         // (Li+1)*(Lo+1)*S
@group(0) @binding(1) var<storage, read> all_ins: array<f32>;          // (Li, S, S)
@group(0) @binding(2) var<storage, read> all_del: array<f32>;          // (Lo, S, S)
@group(0) @binding(3) var<storage, read> all_match: array<f32>;        // (Li, Lo, S, S)
@group(0) @binding(4) var<storage, read> silent: array<f32>;           // (S, S)
@group(0) @binding(5) var<uniform> diag_params: DiagParams;

struct DiagParams {
  d: u32,  // current diagonal
}

const NEG_INF: f32 = -3.402823466e+38f;

fn logaddexp(a: f32, b: f32) -> f32 {
  if (a <= NEG_INF) { return b; }
  if (b <= NEG_INF) { return a; }
  let m = max(a, b);
  return m + log(exp(a - m) + exp(b - m));
}

fn dp_idx(i: u32, o: u32, s: u32) -> u32 {
  return (i * (LO + 1u) + o) * S + s;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let d = diag_params.d;
  let thread_idx = gid.x;

  // Compute (i, o) from diagonal and thread index
  let i_min = select(0u, d - LO, d > LO);
  let i = i_min + thread_idx;
  let o = d - i;

  // Bounds check
  if (i > LI || o > LO) { return; }
  if (i == 0u && o == 0u) { return; }

  let ss = S * S;

  // Initialize cell to NEG_INF
  var cell: array<f32, 64>;  // max 64 states
  for (var s = 0u; s < S; s++) {
    cell[s] = NEG_INF;
  }

  // Match from (i-1, o-1)
  if (i > 0u && o > 0u) {
    let match_base = ((i - 1u) * LO + (o - 1u)) * ss;
    for (var dst = 0u; dst < S; dst++) {
      var acc = NEG_INF;
      for (var src = 0u; src < S; src++) {
        let prev = dp[dp_idx(i - 1u, o - 1u, src)];
        let w = all_match[match_base + src * S + dst];
        acc = logaddexp(acc, prev + w);
      }
      cell[dst] = logaddexp(cell[dst], acc);
    }
  }

  // Insert from (i-1, o)
  if (i > 0u) {
    let ins_base = (i - 1u) * ss;
    for (var dst = 0u; dst < S; dst++) {
      var acc = NEG_INF;
      for (var src = 0u; src < S; src++) {
        let prev = dp[dp_idx(i - 1u, o, src)];
        let w = all_ins[ins_base + src * S + dst];
        acc = logaddexp(acc, prev + w);
      }
      cell[dst] = logaddexp(cell[dst], acc);
    }
  }

  // Delete from (i, o-1)
  if (o > 0u) {
    let del_base = (o - 1u) * ss;
    for (var dst = 0u; dst < S; dst++) {
      var acc = NEG_INF;
      for (var src = 0u; src < S; src++) {
        let prev = dp[dp_idx(i, o - 1u, src)];
        let w = all_del[del_base + src * S + dst];
        acc = logaddexp(acc, prev + w);
      }
      cell[dst] = logaddexp(cell[dst], acc);
    }
  }

  // Propagate silent transitions
  for (var iter = 0u; iter < S; iter++) {
    var changed = false;
    for (var dst = 0u; dst < S; dst++) {
      var acc = NEG_INF;
      for (var src = 0u; src < S; src++) {
        let val = cell[src] + silent[src * S + dst];
        acc = logaddexp(acc, val);
      }
      let new_val = logaddexp(cell[dst], acc);
      // Only sets changed if the original cell value was improved
      // by the initial cell value combined with the accumulation
      if (abs(new_val - cell[dst]) > 1e-7f) {
        cell[dst] = new_val;
        changed = true;
      }
    }
    if (!changed) { break; }
  }

  // Write back
  for (var s = 0u; s < S; s++) {
    dp[dp_idx(i, o, s)] = cell[s];
  }
}
