# Changelog

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
