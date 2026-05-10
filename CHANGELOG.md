# Changelog

## Unreleased

### Changed
- **`--rust` CLI flag renamed to `--rust-phylo-hmm`** for symmetry with the new `--rust-transducer` (regular WFST codegen). Same semantics — multidim Forward / Viterbi DP on a phylo-composed generator. All Makefile targets, tests, and docs updated. Downstream callers using `boss … --rust` need to switch to `--rust-phylo-hmm`.
- **`--tkf92-branch-AAA-MMM` is now a 6-state WFST** `[begin, match, hold, insert, delete, end]` that correctly factors the joint TKF92 pair HMM by the zero-inflated `--tkf92-root-AAA-MMM` singlet. The extra `hold` state distinguishes pre-input from post-input inserts: transitions out of `begin`/`hold` divide by κ (singlet still at start), while transitions out of `match`/`insert`/`delete` divide by ν (singlet in its insert loop). The previous 5-state version did not factor the zero-inflated singlet correctly.

### Added
- **`--evolmoves-root-AAA-MMM`** and **`--evolmoves-branch-AAA-MMM`**: new TKF92 variant whose root is a non-zero-inflated ν-geometric singlet (P(L=0)=1−ν, P(L=k≥1)=ν^k·(1−ν)) and whose branch is the 5-state regularised conditional pair HMM that uniformly divides input-consuming transitions by ν. Composing the two recovers the standard 5-state TKF92 joint pair HMM matrix to ulp; equivalent to composing `--tkf92-root` with `--tkf92-branch` (the new 6-state). Useful when you want a uniform conditional normalisation.

### Removed
- **Standalone `tkf92-branch-prot-f81` preset** (the hand-tuned `preset/tkf92-branch-prot-f81.json` + the `addPresetAs` registration). Superseded by the parameterised `--tkf92-branch-prot-f81` CLI flag, which now emits the canonical 6-state factoring of the zero-inflated TKF92 root singlet. Downstream Python (`test_beam_align.py`) and JavaScript (`test-beam-align.mjs`) tests migrated to invoke the CLI flag via subprocess.

### Added
- New CLI flag `--rust-transducer` (mutually exclusive with `--rust-phylo-hmm`): emits a Rust crate for the standard 2D Forward / Viterbi DP on a regular Machine Boss transducer (string input, string output, no multi-leaf phylo intersection). Crate exposes `forward(p: &Params, input: &[&str], output: &[&str]) -> f64` and (unless `--no-viterbi`) `viterbi(...)`. Bakes the machine JSON; pre-evaluates per-call weights via the WeightAlgebra evaluator (no bytecode VM); silent / match / insert / delete transitions are bucketed at setup. Independent code path from `--rust-phylo-hmm` (compileRust) so the existing phylo-multidim Rust API is unchanged.

### Fixed
- TKF92 root preset (`--tkf92-root-AAA-MMM`): the decide-state loop probability now uses ν = r + (1−r)·κ as specified by the TKF92 ν-modified geometric, instead of `r` directly. Previous output was P(L=k≥1) = κ·r^(k−1)·(1−r); now correctly P(L=k≥1) = κ·ν^(k−1)·(1−ν). The 4-state structure also reshaped to a 3-state `[start, insert, end]` shape so all silent transitions are strictly forward (standalone Forward DP no longer requires composition pre-processing).

### Added
- TKF preset family extended:
  - New substitution kernels for the binary alphabet: `telegraph` (asymmetric 2-state CTMC with `rate01`, `rate10`), `bsc` (Binary Symmetric Channel — symmetric Telegraph with single `flipRate`), `erasure` (Binary Erasure Channel — 0-absorbing Telegraph with `eraseRate`).
  - New version `iid`: zero-indel-rate degeneration of TKF91. `--iid-branch-AAA-MMM` is a memoryless 1:1 channel (single emit state with self-loops, no insert/delete states). `--iid-root-AAA-MMM` is a geometric-length emitter with a free `pExtend` probability (same WFST shape as TKF91 root, but `pExtend` is independent of `insRate`/`delRate`).
  - CLI flag generalized to `--(tkf91|tkf92|iid)-(root|branch)-AAA-MMM` with `MMM ∈ {jc, f81, k80, hky85, id, telegraph, bsc, erasure}`.

## [0.1.0] - 2025-01-01

### Added
- Plan7 HMMER support: full Plan7 architecture with single-hit (`--hmmer-plan7`) and multi-hit (`--hmmer-multihit`) modes
- Python/JAX package (`machineboss`): machine construction, weight algebra, evaluation, Forward/Backward/Viterbi on GPU via JAX
- Fused Plan7+transducer algorithms (GeneWise-style) avoiding explicit state-space composition
- CLI (`boss`): transducer construction, composition, intersection, concatenation, union, Kleene closure
- Forward, Backward, and Viterbi algorithms
- Baum-Welch training via GSL optimizers
- Beam search encoding/decoding
- CTC prefix search, MCMC, simulated annealing decoding
- Code generation (C++/JS) for Forward algorithm
- HMMER profile HMM import
- CSV profile import
- Regular expression construction
- Preset machines (translate, dnapsw, protpsw, Jukes-Cantor, TKF91, Hamming codes, etc.)
- JSON schema validation
- GraphViz DOT output
