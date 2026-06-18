//! Fused Plan7 + transducer Forward/Viterbi kernel (Rust).
//!
//! A faithful, numerically close-to-bit-identical port of Machine Boss's CPU
//! JavaScript reference `js/webgpu/cpu/fused-plan7.mjs` (which is itself the
//! reference for the WebGPU f32 kernel). It scores an output sequence (e.g. DNA)
//! against a protein Plan7 profile HMM *fused* with a protein->output transducer
//! (e.g. a codon + intron model), without materializing the composed state space.
//!
//! # What this crate provides
//!
//! - [`parse_hmmer`] — parse an HMMER3 text profile (self-contained).
//! - [`prepare_machine`] — turn a Machine Boss JSON transducer + params into a
//!   SPARSE log-transition representation (see below).
//! - [`tokenize`] — map a symbol string to 1-based token indices.
//! - [`build_fused_plan7`] — build the fused-machine data (`fm`) once.
//! - [`fused_plan7_forward`] / [`fused_plan7_viterbi`] — run the DP.
//!
//! # Sparse transducer (the key efficiency change vs. the JS reference)
//!
//! The JS CPU reference iterates the DENSE `S_td x S_td` transducer matrix for
//! every emission (`O(L * K * S_td^2)`). For real transducers (e.g. `prot2dna`
//! with ~132 states) that is far too slow. This crate iterates only the
//! transitions that actually exist: per `(in_tok, out_tok)`, a list of
//! `(src, dst, log_weight)` edges (plus the dense silent block, which the
//! fixed-point silent closure genuinely needs dense). Cost becomes
//! `O(L * K * nnz)`. Skipping absent (`NEG_INF`) entries is exactly equivalent
//! to the dense reduce because `NEG_INF` is the identity of both the logsumexp
//! and the max-plus reduce.
//!
//! # Numerics
//!
//! All math is f64. Per-column reduces are performed in the same order
//! (`src` ascending) and with the same two-pass logsumexp formulation as the JS
//! `reduce`, so results match the CPU JS reference to tight tolerance (Forward
//! within ~1e-6, Viterbi essentially exact on the test fixtures).
//!
//! # Wasm
//!
//! The crate builds for `wasm32-unknown-unknown`. The hot DP path uses only
//! `Vec<f64>` scratch and `f64` math (no threads / filesystem). `serde_json` is
//! used only to parse a transducer JSON `Value` up front.
//!
//! # Calling from another Rust crate
//!
//! ```no_run
//! use std::collections::HashMap;
//! use serde_json::Value;
//! use fused_plan7::{
//!     parse_hmmer, prepare_machine, tokenize,
//!     build_fused_plan7, fused_plan7_forward, fused_plan7_viterbi,
//!     FusedOpts, Semiring,
//! };
//!
//! # fn run() -> Result<(), String> {
//! let hmm_text: String = std::fs::read_to_string("profile.hmm").map_err(|e| e.to_string())?;
//! let model = parse_hmmer(&hmm_text)?;
//!
//! let machine_json: Value =
//!     serde_json::from_str(&std::fs::read_to_string("prot2dna.json").map_err(|e| e.to_string())?)
//!         .map_err(|e| e.to_string())?;
//! let params: HashMap<String, f64> = HashMap::new(); // fill codon/flank params
//! let transducer = prepare_machine(&machine_json, &params)?;
//!
//! let fm = build_fused_plan7(&model, &transducer, FusedOpts { multihit: false, l: 400.0 });
//!
//! let dna = tokenize("ATGGCAGATGAA", &transducer.output_alphabet)?;
//! let fwd = fused_plan7_forward(&fm, &dna, Semiring::LogSumExp);
//! let vit = fused_plan7_viterbi(&fm, &dna);
//! println!("forward={fwd} viterbi={vit}");
//! # Ok(())
//! # }
//! ```

pub mod fused;
pub mod hmmer;
pub mod logmath;
pub mod machine_prep;

// ---- Public re-exports (the documented API surface) ----
pub use fused::{build_fused_plan7, fused_plan7_forward, fused_plan7_viterbi, FusedOpts, FusedPlan7};
pub use hmmer::{calc_match_occupancy, parse_hmmer, HmmerModel, HmmerNode};
pub use logmath::{logaddexp, logmax, safe_log, Semiring, NEG_INF};
pub use machine_prep::{
    evaluate_weight, prepare_machine, token_index, tokenize, PreparedMachine, SparseEdge,
};
