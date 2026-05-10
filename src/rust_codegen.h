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

// Regular-transducer mode: emit a Rust crate at outputDir implementing
// the standard 2D Forward / Viterbi DP for a Machine Boss transducer
// with both input and output alphabets — string sequence in, string
// sequence out (no multi-leaf phylo intersection). Use this when you
// have a "regular" WFST you want to call from Rust:
//
//   pub fn forward(p: &Params, input: &[&str], output: &[&str]) -> f64;
//   pub fn viterbi(p: &Params, input: &[&str], output: &[&str]) -> f64;
//
// where input / output are slices of symbol-string references. This is
// the in/out counterpart to compileRust (which is phylo-generator only).
// At minimum one of input or output must have a non-empty alphabet
// (i.e. the machine must have at least one consuming or emitting
// transition).
//
// The implementation bakes the machine JSON as a `&'static str`,
// pre-evaluates per-call weights once via the existing WeightAlgebra
// evaluator (no bytecode VM yet), then runs a straight 2D DP over
// (state, input_index, output_index). Match transitions consume one
// input symbol and emit one output symbol; insert-only consume nothing
// and emit; delete-only consume input but emit nothing; silent
// transitions advance state without consuming or emitting.
void compileRustTransducer (const Machine& m, const std::string& outputDir,
                            bool emitViterbi = true);

// Skeleton-bake mode: emit a minimal Rust crate at outputDir containing
//   pub static T_JSON: &str = ...     // canonical branch transducer (un-renamed)
//   pub static M_SKEL_JSON: &str = ...// skeleton phylo machine
//   pub static TREE_NEWICK: &str = ...// Newick tree string
//   pub static TIME_PARAM: &str = ... // name of T's per-branch time parameter
// plus a stub `prebuild()` that panics. Subsequent increments will replace
// the stub with a Rust port of the WFST algebra (compose / intersect /
// waitingMachine / ergodicMachine / phylo recursion) that consumes the baked
// inputs at startup to materialise the per-symbol M_full tables that the
// existing Forward / Viterbi DP code expects, eliminating boss's role in
// holding M_full in memory during codegen. See task #38.
void compileRustSkeleton (const Machine& M_skel, const Machine& T,
                          const std::string& tree_newick,
                          const std::string& time_param,
                          const std::string& outputDir);

}  // namespace MachineBoss

#endif  // RUST_CODEGEN_INCLUDED
