// Build per-position transfer matrices for 1D prefix scan.
//
// For each position p, computes:
//   M[p] = closure @ emit_trans[token[p]] @ closure
//
// where emit_trans[tok] = logTrans[tok, 0, :, :] (for input) or
//                         logTrans[0, tok, :, :] (for output).
//
// Uniforms provide S (state count), and is_input flag.
// Dispatch: one workgroup per position.

// Constants injected at pipeline creation time via override
override S: u32;             // number of states
override N_TOKENS: u32;      // number of tokens (including null)

@group(0) @binding(0) var<storage, read> tokens: array<u32>;       // (L,) 1-based token indices
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

// mat_mul: C[i][k] = logsumexp_j(A[i][j] + B[j][k])
// A_base, B_base, C_base are offsets into the flat arrays
fn mat_mul_logsumexp(
  A: ptr<function, array<f32, 1024>>,  // scratch, holds A matrix
  B: ptr<function, array<f32, 1024>>,  // scratch, holds B matrix
  C: ptr<function, array<f32, 1024>>,  // output
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
  let L = arrayLength(&tokens);
  if (p >= L) { return; }

  let tok = tokens[p];

  // Load emit matrix for this token: trans_slice[tok, :, :]
  var emit: array<f32, 1024>;
  for (var i = 0u; i < S; i++) {
    for (var j = 0u; j < S; j++) {
      emit[i * S + j] = trans_slice[(tok * S + i) * S + j];
    }
  }

  // Load closure matrix
  var clos: array<f32, 1024>;
  for (var i = 0u; i < S * S; i++) {
    clos[i] = closure[i];
  }

  // M = closure @ emit @ closure
  // Step 1: tmp = emit @ closure
  var tmp: array<f32, 1024>;
  mat_mul_logsumexp(&emit, &clos, &tmp);

  // Step 2: result = closure @ tmp
  var result: array<f32, 1024>;
  mat_mul_logsumexp(&clos, &tmp, &result);

  // Store result
  let base = p * S * S;
  for (var i = 0u; i < S * S; i++) {
    transfers[base + i] = result[i];
  }
}
