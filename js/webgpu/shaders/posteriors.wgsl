// Element-wise posterior computation.
//
// posterior[i, o, s] = exp(fwd[i, o, s] + bwd[i, o, s] - logLikelihood)
//
// Dispatch: one workgroup per cell.

override S: u32;

@group(0) @binding(0) var<storage, read> fwd: array<f32>;              // (Li+1)*(Lo+1)*S
@group(0) @binding(1) var<storage, read> bwd: array<f32>;              // (Li+1)*(Lo+1)*S
@group(0) @binding(2) var<storage, read_write> posteriors: array<f32>; // (Li+1)*(Lo+1)*S
@group(0) @binding(3) var<uniform> post_params: PostParams;

struct PostParams {
  total_cells: u32,   // (Li+1) * (Lo+1)
  log_likelihood: f32,
}

const NEG_INF: f32 = -3.402823466e+38f;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let cell_idx = gid.x;
  if (cell_idx >= post_params.total_cells) { return; }

  let ll = post_params.log_likelihood;
  let base = cell_idx * S;

  for (var s = 0u; s < S; s++) {
    let f = fwd[base + s];
    let b = bwd[base + s];
    let log_post = f + b - ll;
    if (log_post <= NEG_INF) {
      posteriors[base + s] = 0.0;
    } else {
      posteriors[base + s] = exp(log_post);
    }
  }
}
