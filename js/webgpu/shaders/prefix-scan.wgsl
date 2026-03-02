// Hillis-Steele parallel prefix scan for S×S matrices under log-semiring matmul.
//
// Computes inclusive prefix products: prefix[p] = M[0] @ M[1] @ ... @ M[p]
//
// Each pass: M'[i] = M[i - stride] @ M[i]  (for i >= stride), else M'[i] = M[i]
//
// For backward: M'[i] = M[i] @ M[i + stride]  (for i + stride < L)
//
// Dispatch: log2(L) passes, each with L workgroups.
// Double-buffered: alternates between buffers A and B.

override S: u32;             // number of states
override USE_MAX: u32 = 0u;  // 0 = logsumexp, 1 = max (Viterbi)
override IS_BACKWARD: u32 = 0u; // 0 = forward, 1 = backward

@group(0) @binding(0) var<storage, read> input_mats: array<f32>;     // source buffer
@group(0) @binding(1) var<storage, read_write> output_mats: array<f32>; // dest buffer
@group(0) @binding(2) var<uniform> params: ScanParams;

struct ScanParams {
  L: u32,      // number of matrices
  stride: u32, // current stride (1, 2, 4, ...)
}

const NEG_INF: f32 = -3.402823466e+38f;

fn reduce_op(a: f32, b: f32) -> f32 {
  if (USE_MAX != 0u) {
    return max(a, b);
  }
  // logaddexp
  if (a <= NEG_INF) { return b; }
  if (b <= NEG_INF) { return a; }
  let m = max(a, b);
  return m + log(exp(a - m) + exp(b - m));
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let L = params.L;
  let stride = params.stride;

  if (idx >= L) { return; }

  let ss = S * S;
  let out_base = idx * ss;

  if (IS_BACKWARD == 0u) {
    // Forward: prefix[i] = M[i-stride] @ M[i]
    if (idx >= stride) {
      let left_base = (idx - stride) * ss;
      // C[i][k] = reduce_j(left[i][j] + right[j][k])
      for (var i = 0u; i < S; i++) {
        for (var k = 0u; k < S; k++) {
          var acc = NEG_INF;
          for (var j = 0u; j < S; j++) {
            let val = input_mats[left_base + i * S + j] + input_mats[out_base + j * S + k];
            acc = reduce_op(acc, val);
          }
          output_mats[out_base + i * S + k] = acc;
        }
      }
    } else {
      // Copy: no combination needed
      for (var i = 0u; i < ss; i++) {
        output_mats[out_base + i] = input_mats[out_base + i];
      }
    }
  } else {
    // Backward: suffix[i] = M[i] @ M[i+stride]
    if (idx + stride < L) {
      let right_base = (idx + stride) * ss;
      for (var i = 0u; i < S; i++) {
        for (var k = 0u; k < S; k++) {
          var acc = NEG_INF;
          for (var j = 0u; j < S; j++) {
            let val = input_mats[out_base + i * S + j] + input_mats[right_base + j * S + k];
            acc = reduce_op(acc, val);
          }
          output_mats[out_base + i * S + k] = acc;
        }
      }
    } else {
      for (var i = 0u; i < ss; i++) {
        output_mats[out_base + i] = input_mats[out_base + i];
      }
    }
  }
}
