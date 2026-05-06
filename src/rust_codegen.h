#ifndef RUST_CODEGEN_INCLUDED
#define RUST_CODEGEN_INCLUDED

#include <string>
#include "machine.h"

namespace MachineBoss {

// Emit a Rust crate at outputDir implementing a multidimensional Forward
// (and Viterbi) algorithm for the given Machine, which is expected to be a
// generator-style transducer obtained from --phylo-tree on a branch
// transducer with --pair-json output encoding. The number of leaves and
// their order is recovered from the nested structure of the machine's
// pair-token outputs.
//
// The emitted crate exposes:
//   pub struct Params { /* one f64 field per free parameter */ }
//   pub fn forward(p: &Params, leaves: [&[u32]; L]) -> f64
//   pub fn viterbi(p: &Params, leaves: [&[u32]; L]) -> f64   (if emitViterbi)
//
// where L is a compile-time constant baked into the generated code, and
// each leaf is a slice of symbol indices into a generated alphabet table.
//
// Numerical precision: the emitted Rust uses exact log_sum_exp via
// (lo - hi).exp().ln_1p(). The default boss build uses an approximate
// lookup table (LOG_SUM_EXP_LOOKUP_MAX=10) that silently zeros out
// contributions whose log-prob gap exceeds ~10 nats, so for non-trivial
// models the emitted Rust is *more accurate* than `boss -L`. Verification
// tests use a Python exact-lse reference rather than `boss -L`.
void compileRust (const Machine& m, const std::string& outputDir,
                  bool emitViterbi = true);

}  // namespace MachineBoss

#endif  // RUST_CODEGEN_INCLUDED
