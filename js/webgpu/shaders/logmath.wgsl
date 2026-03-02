// Log-space arithmetic for WGSL compute shaders.
// All values are f32 log-probabilities.

const NEG_INF: f32 = -3.402823466e+38f;

// log(exp(a) + exp(b)), numerically stable
fn logaddexp(a: f32, b: f32) -> f32 {
  if (a <= NEG_INF) { return b; }
  if (b <= NEG_INF) { return a; }
  let m = max(a, b);
  return m + log(exp(a - m) + exp(b - m));
}

// Max of two values (Viterbi semiring)
fn logmax(a: f32, b: f32) -> f32 {
  return max(a, b);
}
