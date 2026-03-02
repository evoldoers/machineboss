// Extract forward/backward vectors from prefix/suffix products.
//
// Forward: fwd[p, dst] = logsumexp_src(init[src] + prefix[p, src, dst])
// Backward: bwd[p, src] = logsumexp_dst(suffix[p, src, dst] + term[dst])
//
// Also computes posteriors when both fwd and bwd are provided:
//   posterior[p, s] = exp(fwd[p, s] + bwd[p, s] - logLikelihood)

override S: u32;
override MODE: u32 = 0u;  // 0 = forward extract, 1 = backward extract, 2 = posteriors

@group(0) @binding(0) var<storage, read> prefix_mats: array<f32>;  // (L, S, S) prefix/suffix products
@group(0) @binding(1) var<storage, read> init_or_term: array<f32>; // (S,) init vector (fwd) or term vector (bwd)
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // (L+1, S) extracted vectors
@group(0) @binding(3) var<uniform> extract_params: ExtractParams;

struct ExtractParams {
  L: u32,
  log_likelihood: f32,
}

const NEG_INF: f32 = -3.402823466e+38f;

fn logaddexp(a: f32, b: f32) -> f32 {
  if (a <= NEG_INF) { return b; }
  if (b <= NEG_INF) { return a; }
  let m = max(a, b);
  return m + log(exp(a - m) + exp(b - m));
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.x;
  let L = extract_params.L;

  if (MODE == 0u) {
    // Forward extract
    if (p > L) { return; }

    if (p == 0u) {
      // Position 0: just the initial vector (after closure)
      for (var s = 0u; s < S; s++) {
        output[s] = init_or_term[s];
      }
    } else {
      // fwd[p, dst] = logsumexp_src(init[src] + prefix[p-1, src, dst])
      let mat_base = (p - 1u) * S * S;
      for (var dst = 0u; dst < S; dst++) {
        var acc = NEG_INF;
        for (var src = 0u; src < S; src++) {
          let val = init_or_term[src] + prefix_mats[mat_base + src * S + dst];
          acc = logaddexp(acc, val);
        }
        output[p * S + dst] = acc;
      }
    }
  } else if (MODE == 1u) {
    // Backward extract
    if (p > L) { return; }

    if (p == L) {
      // Terminal position: just the terminal vector (after closure)
      for (var s = 0u; s < S; s++) {
        output[L * S + s] = init_or_term[s];
      }
    } else {
      // bwd[p, src] = logsumexp_dst(suffix[p, src, dst] + term[dst])
      let mat_base = p * S * S;
      for (var src = 0u; src < S; src++) {
        var acc = NEG_INF;
        for (var dst = 0u; dst < S; dst++) {
          let val = prefix_mats[mat_base + src * S + dst] + init_or_term[dst];
          acc = logaddexp(acc, val);
        }
        output[p * S + src] = acc;
      }
    }
  } else {
    // Posteriors: posterior[p, s] = exp(fwd[p, s] + bwd[p, s] - ll)
    // In this mode:
    //   prefix_mats is used as fwd (L+1, S)
    //   init_or_term is used as bwd (L+1, S)
    if (p > L) { return; }
    let ll = extract_params.log_likelihood;
    for (var s = 0u; s < S; s++) {
      let fwd_val = prefix_mats[p * S + s];
      let bwd_val = init_or_term[p * S + s];
      let log_post = fwd_val + bwd_val - ll;
      if (log_post <= NEG_INF) {
        output[p * S + s] = 0.0;
      } else {
        output[p * S + s] = exp(log_post);
      }
    }
  }
}
