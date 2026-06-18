//! Log-space arithmetic utilities.
//!
//! Faithful port of `js/webgpu/internal/logmath.mjs`. All DP values are stored
//! as log-probabilities. `NEG_INF` represents `log(0) = -inf`. The "plus" of the
//! semiring is `logaddexp` (Forward) or `max` (Viterbi); the array "reduce" is
//! the corresponding fold over an iterator of values.
//!
//! Wasm-safe: no allocation, no std I/O, no threads.

/// log(0) = -infinity.
pub const NEG_INF: f64 = f64::NEG_INFINITY;

/// `log(exp(a) + exp(b))`, numerically stable. Matches `logaddexp` in
/// `logmath.mjs` exactly (including the `a === NEG_INF` / `b === NEG_INF`
/// short-circuits, which keep `NEG_INF` an identity element).
#[inline]
pub fn logaddexp(a: f64, b: f64) -> f64 {
    if a == NEG_INF {
        return b;
    }
    if b == NEG_INF {
        return a;
    }
    let m = if a > b { a } else { b };
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// `max(a, b)` — the "plus" of the max-plus (Viterbi) semiring.
#[inline]
pub fn logmax(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}

/// Which semiring a DP runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Semiring {
    /// Forward: plus = logaddexp, reduce = logsumexp.
    LogSumExp,
    /// Viterbi: plus = max, reduce = max.
    MaxPlus,
}

impl Semiring {
    /// Parse the string semiring names used by the JS reference
    /// (`"logsumexp"` / `"maxplus"`).
    pub fn parse_name(s: &str) -> Option<Semiring> {
        match s {
            "logsumexp" => Some(Semiring::LogSumExp),
            "maxplus" => Some(Semiring::MaxPlus),
            _ => None,
        }
    }

    /// Semiring "plus": combine two scalars.
    ///
    /// For `LogSumExp` this is `logaddexp`; for `MaxPlus` this is `max`.
    #[inline]
    pub fn plus(self, a: f64, b: f64) -> f64 {
        match self {
            Semiring::LogSumExp => logaddexp(a, b),
            Semiring::MaxPlus => logmax(a, b),
        }
    }

    /// Semiring "reduce" over an iterator of values.
    ///
    /// Mirrors the JS `reduce(arr)`:
    ///  - max-plus: the maximum (NEG_INF if empty).
    ///  - logsumexp: `m + log(sum exp(x_i - m))` over finite entries, where `m`
    ///    is the max; returns NEG_INF if the max is NEG_INF.
    ///
    /// IMPORTANT: this performs a single max scan then a single sum scan, in the
    /// same order the JS does, so the float64 result is bit-identical for a given
    /// iteration order. Callers must pass values in the same order as the JS
    /// (`src` ascending) to stay bit-identical.
    #[inline]
    pub fn reduce<I>(self, iter: I) -> f64
    where
        I: IntoIterator<Item = f64> + Clone,
    {
        match self {
            Semiring::MaxPlus => {
                let mut m = NEG_INF;
                for x in iter {
                    if x > m {
                        m = x;
                    }
                }
                m
            }
            Semiring::LogSumExp => {
                let mut m = NEG_INF;
                for x in iter.clone() {
                    if x > m {
                        m = x;
                    }
                }
                if m == NEG_INF {
                    return NEG_INF;
                }
                let mut s = 0.0_f64;
                for x in iter {
                    if x != NEG_INF {
                        s += (x - m).exp();
                    }
                }
                m + s.ln()
            }
        }
    }
}

/// `log(x)` with `log(0) = NEG_INF` (the JS `safeLog` / `strToProb`+log idiom).
#[inline]
pub fn safe_log(x: f64) -> f64 {
    if x > 0.0 {
        x.ln()
    } else {
        NEG_INF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg_inf_is_identity() {
        assert_eq!(logaddexp(NEG_INF, -1.5), -1.5);
        assert_eq!(logaddexp(-1.5, NEG_INF), -1.5);
        assert_eq!(logaddexp(NEG_INF, NEG_INF), NEG_INF);
        assert_eq!(logmax(NEG_INF, -2.0), -2.0);
    }

    #[test]
    fn logaddexp_matches_definition() {
        // log(exp(a)+exp(b)) computed naively for small finite values.
        for &(a, b) in &[(-1.0_f64, -2.0_f64), (0.0, 0.0), (-10.0, -0.1), (3.0, -3.0)] {
            let naive = (a.exp() + b.exp()).ln();
            assert!((logaddexp(a, b) - naive).abs() < 1e-12, "a={a} b={b}");
        }
    }

    #[test]
    fn reduce_semirings() {
        let xs = [-1.0_f64, -3.0, NEG_INF, -2.0];
        assert_eq!(Semiring::MaxPlus.reduce(xs.iter().copied()), -1.0);
        let lse = Semiring::LogSumExp.reduce(xs.iter().copied());
        let naive = ((-1.0_f64).exp() + (-3.0_f64).exp() + (-2.0_f64).exp()).ln();
        assert!((lse - naive).abs() < 1e-12);
        // Empty / all-NEG_INF reduce to NEG_INF in both semirings.
        assert_eq!(Semiring::LogSumExp.reduce(std::iter::empty()), NEG_INF);
        assert_eq!(
            Semiring::LogSumExp.reduce([NEG_INF, NEG_INF].iter().copied()),
            NEG_INF
        );
        assert_eq!(Semiring::MaxPlus.reduce(std::iter::empty()), NEG_INF);
    }

    #[test]
    fn semiring_parse_name() {
        assert_eq!(Semiring::parse_name("logsumexp"), Some(Semiring::LogSumExp));
        assert_eq!(Semiring::parse_name("maxplus"), Some(Semiring::MaxPlus));
        assert_eq!(Semiring::parse_name("nope"), None);
    }
}
