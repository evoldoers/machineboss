// 2D Viterbi wavefront kernel with argmax tracking.
//
// Same structure as forward wavefront, but uses max instead of logaddexp
// and stores argmax (source state + move type) for traceback.

override S: u32;
override LI: u32;
override LO: u32;

@group(0) @binding(0) var<storage, read_write> dp: array<f32>;         // (Li+1)*(Lo+1)*S
@group(0) @binding(1) var<storage, read> all_ins: array<f32>;          // (Li, S, S)
@group(0) @binding(2) var<storage, read> all_del: array<f32>;          // (Lo, S, S)
@group(0) @binding(3) var<storage, read> all_match: array<f32>;        // (Li, Lo, S, S)
@group(0) @binding(4) var<storage, read> silent: array<f32>;           // (S, S)
@group(0) @binding(5) var<uniform> diag_params: DiagParams;
@group(0) @binding(6) var<storage, read_write> tb_src: array<i32>;     // traceback source state
@group(0) @binding(7) var<storage, read_write> tb_type: array<u32>;    // traceback move type

struct DiagParams {
  d: u32,
}

const NEG_INF: f32 = -3.402823466e+38f;
const TB_NONE: u32 = 0u;
const TB_MATCH: u32 = 1u;
const TB_INSERT: u32 = 2u;
const TB_DELETE: u32 = 3u;
const TB_SILENT: u32 = 4u;

fn dp_idx(i: u32, o: u32, s: u32) -> u32 {
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
  if (i == 0u && o == 0u) { return; }

  let ss = S * S;

  var cell: array<f32, 64>;
  var best_src: array<i32, 64>;
  var best_type: array<u32, 64>;
  for (var s = 0u; s < S; s++) {
    cell[s] = NEG_INF;
    best_src[s] = -1;
    best_type[s] = TB_NONE;
  }

  // Match from (i-1, o-1)
  if (i > 0u && o > 0u) {
    let match_base = ((i - 1u) * LO + (o - 1u)) * ss;
    for (var dst = 0u; dst < S; dst++) {
      for (var src = 0u; src < S; src++) {
        let prev = dp[dp_idx(i - 1u, o - 1u, src)];
        let w = all_match[match_base + src * S + dst];
        let val = prev + w;
        if (val > cell[dst]) {
          cell[dst] = val;
          best_src[dst] = i32(src);
          best_type[dst] = TB_MATCH;
        }
      }
    }
  }

  // Insert from (i-1, o)
  if (i > 0u) {
    let ins_base = (i - 1u) * ss;
    for (var dst = 0u; dst < S; dst++) {
      for (var src = 0u; src < S; src++) {
        let prev = dp[dp_idx(i - 1u, o, src)];
        let w = all_ins[ins_base + src * S + dst];
        let val = prev + w;
        if (val > cell[dst]) {
          cell[dst] = val;
          best_src[dst] = i32(src);
          best_type[dst] = TB_INSERT;
        }
      }
    }
  }

  // Delete from (i, o-1)
  if (o > 0u) {
    let del_base = (o - 1u) * ss;
    for (var dst = 0u; dst < S; dst++) {
      for (var src = 0u; src < S; src++) {
        let prev = dp[dp_idx(i, o - 1u, src)];
        let w = all_del[del_base + src * S + dst];
        let val = prev + w;
        if (val > cell[dst]) {
          cell[dst] = val;
          best_src[dst] = i32(src);
          best_type[dst] = TB_DELETE;
        }
      }
    }
  }

  // Propagate silent with argmax
  for (var iter = 0u; iter < S; iter++) {
    var changed = false;
    for (var dst = 0u; dst < S; dst++) {
      for (var src = 0u; src < S; src++) {
        let val = cell[src] + silent[src * S + dst];
        if (val > cell[dst]) {
          cell[dst] = val;
          best_src[dst] = i32(src);
          best_type[dst] = TB_SILENT;
          changed = true;
        }
      }
    }
    if (!changed) { break; }
  }

  // Write back
  for (var s = 0u; s < S; s++) {
    let idx = dp_idx(i, o, s);
    dp[idx] = cell[s];
    tb_src[idx] = best_src[s];
    tb_type[idx] = best_type[s];
  }
}
