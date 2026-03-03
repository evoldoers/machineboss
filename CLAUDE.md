# Machine Boss

Bioinformatics automata toolkit for constructing, manipulating, and applying weighted finite-state transducers (WFSTs). Provides both a CLI (`boss`) and C++ library.

## Build

```bash
brew install gsl boost htslib pkgconfig   # macOS deps
make                                       # builds bin/boss
npm install                                # needed for tests
make test                                  # runs full test suite
make clean                                 # removes build artifacts
```

Compiler: clang++ (preferred) or g++, C++11. Links against GSL, Boost (regex, program_options), and zlib.

## Project Layout

- `src/` — C++ source files (core library + headers)
- `target/boss.cpp` — main entry point for the `boss` CLI
- `ext/` — vendored dependencies (nlohmann_json, valijson, cpp-peglib, cpp-httplib, htslib/kseq, fast5, compat)
- `schema/` — JSON Schema files for the transducer format and related data structures
- `preset/` — preset machine JSON files (translate, dnapsw, protpsw, etc.)
- `data/` — parameter data files (codon tables, substitution matrices)
- `constraints/` — parameter constraint files
- `params/` — parameter files
- `js/` — Node.js scripts for generating preset machines and test utilities
- `t/` — test data, expected outputs, and test source files
- `bin/` — compiled binary output (generated)
- `obj/` — compiled object files (generated)
- `wasm/` — WebAssembly build artifacts
- `emcc/` — Emscripten support files
- `python/` — Python utilities (hamming74.py, mixradar.py)
- `img/` — images for documentation
- `examples/` — example scripts

## Architecture

The core types are in `src/machine.h` (Machine, MachineState, MachineTransition) and `src/weight.h` (WeightExpr, weight algebra). The machine JSON format uses an expression language for weights (arithmetic, log, exp, parameters).

Key modules:
- `machine.cpp` — machine construction, composition, intersection, concatenation, union, sort, eliminate
- `forward.cpp/backward.cpp` — Forward/Backward algorithms
- `viterbi.cpp` — Viterbi algorithm
- `beam.cpp` — beam search encoding/decoding
- `ctc.cpp` — CTC prefix search, MCMC, simulated annealing
- `compiler.cpp` — code generation (C++/JS) for Forward algorithm
- `fitter.cpp/counts.cpp` — Baum-Welch training via GSL optimizers
- `eval.cpp` — weight expression evaluation
- `parsers.cpp` — regex, weight expression, and command-line parsing
- `hmmer.cpp` — HMMER profile HMM import
- `csv.cpp` — CSV profile import
- `fastseq.cpp` — FASTA I/O
- `schema.cpp` — JSON schema validation (via valijson)
- `preset.cpp` — built-in preset machines (embedded as xxd includes from `src/preset/`)

## README

The README help text is auto-generated from `boss -h`. Run `make README.md` to update it after changing command-line options.

## Testing

Tests are defined in the Makefile and use `t/testexpect.py` as a harness. Test categories: schema validation, composition, construction, I/O, algebra, dynamic programming, code generation, encoding/decoding, expression parsing, JSON API operations, preset loading. Some tests require `node` (JS tests).

## JSON Format

The native machine format is a restricted JSON representation of a WFST. Schemas are in `schema/`. The start state is always first; the end state is always last. Transitions can use algebraic weight expressions with named parameters.

## Documentation

Documentation is hosted at [machineboss.org](https://machineboss.org/) via GitHub Pages. Markdown source files are in `docs/`:

- `docs/machineboss.md` — program reference
- `docs/json-format.md` — JSON format reference
- `docs/expressions.md` — weight expression mini-language (grammar in `src/grammars/expr.h`, parser in `src/parsers.cpp`)
- `docs/json-output.md` — JSON output format reference (machine, parameters, loglike, alignment, counts, encode, decode)
- `docs/composition.md` — transducer composition algorithm documentation
- `docs/webgpu.md` — WebGPU API reference
