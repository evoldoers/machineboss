# Changelog

## Unreleased

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
