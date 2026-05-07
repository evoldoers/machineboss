---
title: Rust Codegen (Multidim Forward DP)
nav_order: 8.5
permalink: /rust-codegen/
---

# Rust Codegen — multidimensional Forward / Viterbi for phylo composition

The `--codegen DIR --rust` option emits a self-contained Rust crate
implementing a multidimensional Forward (and Viterbi) dynamic programming
algorithm specialised to whatever generator transducer is on the stack.

The intended use case is a phylogenetic composition of a branch transducer
on a tree: e.g. **TKF92 + HKY85 on `(A,B,(C,D)Y)X;`**, with the leaves
observed and the per-branch parameters exposed at the Rust API level.

## What gets emitted

```
$DIR/
├── Cargo.toml
└── src/
    └── lib.rs
```

`lib.rs` exposes:

```rust
pub const NUM_LEAVES: usize = …;       // recovered from pair-token nesting
pub const NUM_STATES: usize = …;
pub const ALPHABET: [&str; …] = […];   // leaf alphabet, in codegen order

pub struct Params {
    pub insRate:  f64,                 // λ
    pub delRate:  f64,                 // μ
    pub r:        f64,                 // TKF92 geometric extension
    pub tsRatio:  f64,                 // HKY85 κ
    pub pi_A:     f64,
    pub pi_C:     f64,
    pub pi_G:     f64,
    pub pi_T:     f64,
    pub t_A_:     f64,                 // branch length X→A
    pub t_B_:     f64,                 //               X→B
    pub t_Y_:     f64,                 //               X→Y
    pub t_C_:     f64,                 //               Y→C
    pub t_D_:     f64,                 //               Y→D
}

pub fn forward(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> f64;
pub fn viterbi(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> f64;
```

Each `leaves[i]` is a slice of symbol indices into `ALPHABET`. For DNA the
mapping is `A=0, C=1, G=2, T=3` (sorted alphabetically).

## Numerical precision

The emitted Rust uses exact `(lo - hi).exp().ln_1p()` for log-sum-exp.
The default `boss` build uses an approximate lookup table
(`LOG_SUM_EXP_LOOKUP_MAX = 10`) that silently zeros out contributions whose
log-prob gap exceeds ~10 nats; for non-trivial models the emitted Rust is
**more accurate** than `boss -L`. The verification tests (`check_tkf91.py`,
`check_tkf92_triad.py`) confirm agreement with an independent exact-lse
Python implementation to floating-point precision (~1e-15).

## How the codegen scales

The naive approach of inlining each weight expression as Rust code is
quadratic in model complexity (geomsum reductions of silent cycles produce
extremely long shared sub-expressions) and a quartet on TKF92+HKY85
overwhelms `rustc` — the emitted source approaches 1 GB.

To keep things compilable, the codegen uses a small register-machine
representation:

  - all unique sub-expressions across all weights are CSE'd globally
  - each unique node becomes one entry in a flat `&[u8]` opcode table
    plus parallel `&[u32]` arg tables (`VM_OPCODES`, `VM_ARG_A`, `VM_ARG_B`)
  - `compute_log_weights` is a single straight-line interpreter loop;
    the bulk of the emitted source is data, not code.

`rustc` compiles huge static arrays linearly, so this scales: a quartet
goes from "doesn't compile" to ~50 s.

## End-to-end recipe — TKF92 + HKY85 quartet

```bash
# 1) Build boss
make

# 2) Codegen the open phylogenetic composition.
#    --pair-json is required so output tokens are JSON-decodable (the codegen
#    recovers the tree topology + leaf count from the nested pair-token shape).
bin/boss \
  --pair-json \
  --tkf92-root-dna-hky85 \
  -m \
  --begin --tkf92-branch-dna-hky85 \
  --phylo-tree-string '(A,B,(C,D)Y)X;' \
  --end \
  --codegen path/to/quartet --rust
```

This emits `path/to/quartet/Cargo.toml` and `path/to/quartet/src/lib.rs`.
For the quartet the codegen takes ~18 s and the resulting crate is ~40 MB.

```bash
# 3) Compile (release).
cd path/to/quartet
cargo build --release
```

`cargo build --release` for the quartet takes ~50 s on a recent macOS x86_64.

```rust
// 4) Use it. Drop a binary into path/to/quartet/examples/run.rs:
use phylo_dp::{forward, viterbi, Params, ALPHABET};

fn idx(c: char) -> u32 {
    ALPHABET.iter().position(|x| x.chars().next() == Some(c)).unwrap() as u32
}

fn main() {
    let p = Params {
        // TKF92 indel parameters
        insRate: 0.05,   // λ
        delRate: 0.06,   // μ
        r:       0.4,    // geometric extension
        // HKY85 substitution parameters
        tsRatio: 2.0,
        pi_A: 0.30, pi_C: 0.20, pi_G: 0.25, pi_T: 0.25,
        // Branch lengths (per-branch time parameters, named after child node)
        t_A_: 0.10,  // X → A
        t_B_: 0.20,  // X → B
        t_Y_: 0.05,  // X → Y
        t_C_: 0.15,  // Y → C
        t_D_: 0.18,  // Y → D
    };
    let a: Vec<u32> = "ACGT".chars().map(idx).collect();
    let b: Vec<u32> = "ACG" .chars().map(idx).collect();
    let c: Vec<u32> = "ACT" .chars().map(idx).collect();
    let d: Vec<u32> = "AGT" .chars().map(idx).collect();
    let f = forward(&p, [&a, &b, &c, &d]);
    let v = viterbi(&p, [&a, &b, &c, &d]);
    println!("forward log-likelihood = {}", f);
    println!("viterbi log-likelihood = {}", v);
}
```

```bash
cargo run --release --example run
```

## Performance — quartet length sweep

Wall-clock per call to `forward` / `viterbi` on the (A,B,(C,D)Y)X; tree
with TKF92+HKY85, leaves of equal length n (release build, macOS x86\_64):

| n  | forward (ms) | viterbi (ms) |
| -- | ------------ | ------------ |
| 2  |       38     |       18     |
| 3  |       87     |       66     |
| 5  |      550     |      410     |
| 8  |     3983     |     2655     |
| 10 |     8647     |     6606     |

Scaling is O(L⁴) — DP grid is (n+1)⁴ cells × 1249 states × 211 248 emitting
transitions per cell-update. Two structural optimizations apply:

  - **Delta-vector sharding** — emitting transitions are grouped by their
    per-leaf delta vector, so feasibility checks (does the predecessor
    cell exist?) hoist out of the per-transition loop. This gives Viterbi
    a ~60% speedup over a naive unsharded version (since `max` is cheap
    enough that the predicate cost dominates) and Forward a ~25% speedup.
  - **lse cutoff** — `log_sum_exp(a, b)` returns `max(a, b)` exactly when
    `|a - b| > 36` nats, since `exp(-36) ≈ 2.3e-16` is below f64's
    relative epsilon and adds nothing observable. This skips two
    transcendental calls (`exp` and `ln_1p`) for negligible contributions
    and gives Forward another ~16% speedup.

One-time costs for the quartet:
  - codegen:  ~18 s  (boss writes the crate)
  - rustc:    ~36 s  (release build with `lto = true`)

## Verification

  - `make test-rust-codegen-echo` — trivial echo branch transducer; codegen
    Forward matches `boss --phylo-clamp -L` to floating-point precision.
  - `make test-rust-codegen-tkf91` — TKF91 + JC on `(A,B)P;`. Matches an
    exact-lse Python Forward over the clamped machine to ~1e-15.
  - `make test-rust-codegen-tkf92-triad` *(manual, not in default test set)* —
    TKF92 + HKY85 on `(A,B,C)X;`. Full Forward reference comparison via
    Python multidim DP. Matches to ~1e-15.
  - `make test-rust-codegen-tkf92-quartet` *(manual, not in default test set)* —
    TKF92 + HKY85 on `(A,B,(C,D)Y)X;`. End-to-end smoke test plus the length
    sweep table above.

## Notes / caveats

  - **`--pair-json` is required** at codegen time. The codegen recovers the
    tree topology and leaf count from the nested structure of pair tokens,
    and JSON makes that structure unambiguous.
  - **Generator transducers only.** The top-of-stack machine must have an
    empty input alphabet. For TKF92, compose a root generator (e.g.
    `--tkf92-root-dna-hky85`) with the phylogenetic intersection of a
    branch transducer.
  - **Leaf order** is determined by left-to-right traversal of the tree.
    For `(A,B,(C,D)Y)X;` that's `A, B, C, D` — `leaves[0]` is A, etc.
  - **`pi_*` constraints**: HKY85 stationary frequencies must sum to 1.
    The Rust crate does not enforce this; the caller is responsible.
