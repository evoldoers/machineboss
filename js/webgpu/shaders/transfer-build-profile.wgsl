// Build per-position transfer matrices for 1D prefix scan (PSWM profile version).
//
// For each position p, computes:
//   emit[src][dst] = logsumexp_k(profile[p, k] + trans_slice[k+1, src, dst])
//   M[p] = closure @ emit @ closure
//
// Unlike the tokenized version which selects one token per position,
// this combines all emitting tokens weighted by log-profile weights.
//
// Dispatch: one workgroup per position.

override S: u32;             // number of states
override N_TOKENS: u32;      // number of tokens (including epsilon at index 0)
override N_ALPHA: u32;       // number of emitting symbols (N_TOKENS - 1)

@group(0) @binding(0) var<storage, read> profile: array<f32>;      // (L, N_ALPHA) log-profile
@group(0) @binding(1) var<storage, read> trans_slice: array<f32>;  // (N_TOKENS, S, S) transition matrices
@group(0) @binding(2) var<storage, read> closure: array<f32>;      // (S, S) closure matrix
@group(0) @binding(3) var<storage, read_write> transfers: array<f32>; // (L, S, S) output transfer matrices

const NEG_INF: f32 = -3.402823466e+38f;

fn logaddexp(a: f32, b: f32) -> f32 {
  if (a <= NEG_INF) { return b; }
  if (b <= NEG_INF) { return a; }
  let m = max(a, b);
  return m + log(exp(a - m) + exp(b - m));
}

fn mat_mul_logsumexp(
  A: ptr<function, array<f32, 1024>>,
  B: ptr<function, array<f32, 1024>>,
  C: ptr<function, array<f32, 1024>>,
) {
  for (var i = 0u; i < S; i++) {
    for (var k = 0u; k < S; k++) {
      var acc = NEG_INF;
      for (var j = 0u; j < S; j++) {
        let val = (*A)[i * S + j] + (*B)[j * S + k];
        acc = logaddexp(acc, val);
      }
      (*C)[i * S + k] = acc;
    }
  }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.x;
  let L = arrayLength(&profile) / N_ALPHA;
  if (p >= L) { return; }

  // Build emission matrix by combining all emitting tokens with profile weights
  var emit: array<f32, 1024>;
  for (var i = 0u; i < S * S; i++) {
    emit[i] = NEG_INF;
  }

  for (var k = 0u; k < N_ALPHA; k++) {
    let tok = k + 1u;  // skip epsilon (index 0)
    let log_weight = profile[p * N_ALPHA + k];
    if (log_weight > NEG_INF) {
      for (var src = 0u; src < S; src++) {
        for (var dst = 0u; dst < S; dst++) {
          let val = log_weight + trans_slice[(tok * S + src) * S + dst];
          emit[src * S + dst] = logaddexp(emit[src * S + dst], val);
        }
      }
    }
  }

  // Load closure matrix
  var clos: array<f32, 1024>;
  for (var i = 0u; i < S * S; i++) {
    clos[i] = closure[i];
  }

  // M = closure @ emit @ closure
  var tmp: array<f32, 1024>;
  mat_mul_logsumexp(&emit, &clos, &tmp);

  var result: array<f32, 1024>;
  mat_mul_logsumexp(&clos, &tmp, &result);

  // Store result
  let base = p * S * S;
  for (var i = 0u; i < S * S; i++) {
    transfers[base + i] = result[i];
  }
}
