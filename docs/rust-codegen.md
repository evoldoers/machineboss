---
title: Rust Codegen (Multidim Forward DP)
nav_order: 8.5
permalink: /rust-codegen/
---

# Rust Codegen — multidimensional Forward / Viterbi for phylo composition

Two emission methods are available, both invoked via `--codegen DIR --rust`.
They differ in what the emitted crate bakes in vs. computes at startup:

  1. **Straight-line codegen** *(default)* — boss runs the full WFST algebra
     (`compose`, `intersect`, `waitingMachine`, `ergodicMachine`,
     `advanceSort`, `processCycles`, plus the buildSubtree recursion) at
     codegen time, then emits the resulting `M_full` as a flat opcode
     bytecode VM (`VM_OPCODES`, `VM_ARG_A`, …) plus straight-line Forward /
     Viterbi loops over delta-vector-sharded emit transitions. The crate
     compiles in ~50 s for a TKF92+HKY85 quartet and runs at the
     wall-clock numbers in the [length sweep table](#performance--quartet-length-sweep) below.
     This is what you get from `bin/boss --pair-json … --codegen DIR --rust`.

  2. **Baked-in unary skeleton, expanded from Rust**
     *(`--phylo-skeleton --codegen DIR --rust`)* — boss bakes the
     **branch transducer T** (un-renamed), the **unary phylo skeleton
     `M_skel`** (same accessible-state set as `M_full` but with placeholder
     emit transitions and chain-collapsed silent transitions), the
     **Newick tree string**, and the **time-parameter name** as four
     `&'static str` constants. The emitted Rust crate carries Rust ports
     of the WFST algebra and runs `phylo_intersect(T, tree, time_param)`
     at startup (in `prebuild()`) to materialise `M_full` from the bake.
     A multidim Forward DP (`forward.rs`) consumes that machine to compute
     the log-likelihood. This shifts work from codegen-time to crate
     startup, and keeps the emitted source small even for wide-alphabet
     models: the bake is dominated by `M_SKEL_JSON` (which is alphabet-
     independent) and `T_JSON` (5–7 states); the
     [skeleton optimisation](/phylogeny/#unary-skeleton) is what makes
     this tractable.
     See [§Skeleton-bake mode](#skeleton-bake-mode) below.

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
pub fn backward(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> f64;
pub fn viterbi(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> f64;

// Full Forward / Backward DP matrices (for posterior calculations).
pub fn forward_matrix(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> DPMatrix;
pub fn backward_matrix(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> DPMatrix;

// Forward-Backward expected transition counts (one entry per bucketed
// transition; bucket order = silents first, then emittings in shard
// order).
pub fn forward_backward_counts(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> FBResult;

// Per-cell posteriors. `f` and `b` come from forward_matrix/backward_matrix.
pub fn state_log_posterior(f: &DPMatrix, b: &DPMatrix,
                           state: u32, idx: [usize; NUM_LEAVES]) -> f64;
pub fn silent_transition_log_posterior(f: &DPMatrix, b: &DPMatrix, lw: &[f64],
                                       bucket: usize, idx: [usize; NUM_LEAVES]) -> f64;
pub fn emitting_transition_log_posterior(f: &DPMatrix, b: &DPMatrix,
                                         leaves: [&[u32]; NUM_LEAVES], lw: &[f64],
                                         shard: usize, i: usize,
                                         idx: [usize; NUM_LEAVES]) -> f64;

// Amortized forms — call precompute_log_weights once, reuse for many
// `*_with_log_weights` calls.
pub fn precompute_log_weights(p: &Params) -> Vec<f64>;
pub fn forward_with_log_weights(lw: &[f64], leaves: ...) -> f64;
pub fn backward_with_log_weights(lw: &[f64], leaves: ...) -> f64;
pub fn viterbi_with_log_weights(lw: &[f64], leaves: ...) -> f64;
pub fn forward_matrix_with_log_weights(lw: &[f64], leaves: ...) -> DPMatrix;
pub fn backward_matrix_with_log_weights(lw: &[f64], leaves: ...) -> DPMatrix;
pub fn forward_backward_counts_with_log_weights(lw: &[f64], leaves: ...) -> FBResult;
```

The codegen also writes a `machine.json` next to `Cargo.toml`. This is
the bucketed composed machine in standard Machine Boss JSON shape, with
each transition carrying an `__C<idx>__` placeholder in its
`expected_count` field. The Rust crate embeds it via `include_str!` and
`FBResult::to_machine_json(&mut out)` substitutes the placeholders with
numeric counts:

```rust
let fb = forward_backward_counts(&p, [&a, &b]);
let mut s = String::new();
fb.to_machine_json(&mut s);   // s is now valid JSON, parseable
println!("{}", s);
```

Each `leaves[i]` is a slice of symbol indices into `ALPHABET`. For DNA the
mapping is `A=0, C=1, G=2, T=3` (sorted alphabetically).

The two-step API is useful when running many DP calls under the same
parameter set (e.g. a length sweep, a benchmark, or sampling many leaf
configurations). `precompute_log_weights` runs the bytecode VM once and
returns a flat `Vec<f64>` of all log-weights; `forward_with_log_weights`
then skips that prelude and only does the cell loop. On the TKF92 quartet
the prelude is ~6 ms — negligible at long sequence lengths but a
worthwhile saving across many short calls.

## Numerical precision

The emitted Rust uses exact `(lo - hi).exp().ln_1p()` for log-sum-exp.
The default `boss` build uses an approximate lookup table
(`LOG_SUM_EXP_LOOKUP_MAX = 10`) that silently zeros out contributions whose
log-prob gap exceeds ~10 nats; for non-trivial models the emitted Rust is
**more accurate** than `boss -L`. The verification tests (`check_tkf91.py`,
`check_tkf92_triad.py`) compare against an independent exact-lse Python
implementation. We observe agreement to floating-point noise (`|fwd-ref|`
in the 1e-15 range on macOS x86_64) and the tests enforce a 1e-12
tolerance to leave headroom for platform variation in the order of
floating-point operations.

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

## Posterior probabilities

Once you have run Forward and Backward, three classes of posterior are
straightforward to read off the matrices.

### The arithmetic

Let `f[s, c]` be the Forward log-prob of being in state `s` at multi-cell
`c = (i_0, …, i_{L-1})` and `b[s, c]` the Backward log-prob of completing
the model from there to the end. Both are populated by `forward_matrix`
and `backward_matrix`, with `f.log_likelihood == b.log_likelihood` to
floating-point noise — call this scalar `Z` (the partition function).
All posteriors below are obtained from `(f, b, Z)` with no further DP.

| Quantity | Formula |
|---|---|
| State posterior at cell `c`, state `s` | `exp(f[s,c] + b[s,c] − Z)` |
| Silent transition `(src→dst, w)` firing at cell `c` | `exp(f[src,c] + log w + b[dst,c] − Z)` |
| Emitting bucket `(src→dst, δ, σ, w)` firing at cell `c` | `exp(f[src, c−δ] + log w + b[dst,c] − Z)` *(when feasible: predecessor cell exists and observed[k][i_k − 1] == σ_k for each k with δ_k = 1)* |
| Expected count of bucket `t` | `Σ_c P(t fires at c) = exp(lse_c(...) − Z)` |

The codegen exposes one helper per row, all returning the **log** of
the quantity (`f64::NEG_INFINITY` for an infeasible transition):

```rust
state_log_posterior(f, b, state, idx)                            // row 1
silent_transition_log_posterior(f, b, lw, bucket, idx)           // row 2
emitting_transition_log_posterior(f, b, leaves, lw, shard, i, idx) // row 3
forward_backward_counts(p, leaves) -> FBResult                   // row 4
```

`bucket` indexes into `SILENT_TRANSITIONS` (range `0..NUM_SILENT`).
Emitting transitions are addressed by `(shard, i)` where `shard` is
the delta-vector index `0..NUM_DELTA` and `i` is the offset within that
shard. The global bucket index used by `FBResult::expected_counts` is
`bucket` for silents, `NUM_SILENT + EMIT_SHARD_BUCKET_OFFSET[shard] + i`
for emittings (or call `emit_bucket_index(shard, i)`).

### State posteriors

```rust
use phylo_dp::{forward_matrix, backward_matrix, state_log_posterior,
               Params, NUM_STATES, NUM_LEAVES};

let p = Params { /* ... */ };
let leaves = [&a[..], &b[..]];
let f  = forward_matrix(&p, leaves);
let bm = backward_matrix(&p, leaves);

// Marginal probability of passing through each state at multi-index idx:
let idx = [3usize, 2usize];     // example: 3 chars consumed at A, 2 at B
let mut marginal = vec![0.0; NUM_STATES];
for s in 0..NUM_STATES {
    let lp = state_log_posterior(&f, &bm, s as u32, idx);
    marginal[s] = if lp.is_finite() { lp.exp() } else { 0.0 };
}
// Sanity: at the origin and final cells the marginal sums to 1
// (start state always reached at origin, end state at full).
```

A useful sanity check (and a default-set regression test):
`Σ_s state_log_posterior(f, b, s, [0;L]).exp() == 1` and the same at
`idx = [len_0, …, len_{L-1}]`. The check_forward_backward.py test asserts
both within `1e-9`.

### Transition posteriors

```rust
use phylo_dp::{
    SILENT_TRANSITIONS, EMIT_SHARDS, DELTA_VEC, NUM_DELTA,
    silent_transition_log_posterior, emitting_transition_log_posterior,
    precompute_log_weights,
};

let lw = precompute_log_weights(&p);

// Marginal probability that a *specific* silent transition fires somewhere
// in the alignment, marginalised over alignment cells.
fn silent_total(p_idx: usize, fm: &DPMatrix, bm: &DPMatrix, lw: &[f64], lens: [usize; NUM_LEAVES]) -> f64 {
    let mut idx = [0usize; NUM_LEAVES];
    let mut total = 0.0;
    loop {
        let lp = silent_transition_log_posterior(fm, bm, lw, p_idx, idx);
        if lp.is_finite() { total += lp.exp(); }
        // advance idx through every cell in lex order ...
        let mut k = NUM_LEAVES;
        let mut advanced = false;
        while k > 0 {
            k -= 1;
            if idx[k] < lens[k] { idx[k] += 1; for j in (k+1)..NUM_LEAVES { idx[j] = 0; } advanced = true; break; }
        }
        if !advanced { break; }
    }
    total
}
```

This `silent_total` is exactly `expected_counts[p_idx]`, so in practice you
should call `forward_backward_counts(&p, leaves)` and read the value out.
The per-cell helper is only worth using when you need a position-specific
posterior — e.g. a heatmap of "where in the alignment does insertion
happen most?".

For an emitting bucket, the per-cell posterior captures column-specific
information about *where* a particular emission profile fires:

```rust
let shard = 5; let i = 17;            // e.g. some specific (delta, syms) pair
let idx   = [4usize, 3usize, 4usize]; // post-emission cell
let lp = emitting_transition_log_posterior(&f, &bm, leaves, &lw, shard, i, idx);
let p  = if lp.is_finite() { lp.exp() } else { 0.0 };
// `p` is the probability that this bucket consumed the leaf characters
// observed[k][idx[k] - 1] for each k where DELTA_VEC[shard][k] == 1,
// at this particular alignment cell.
```

### Expected transition counts and `machine.json`

For aggregate per-transition counts (summed over all alignment cells),
use `forward_backward_counts`:

```rust
let fb = forward_backward_counts(&p, leaves);
println!("log P(observed leaves) = {}", fb.log_likelihood);
println!("counts.len() = {}", fb.expected_counts.len());

// Render the bucketed machine + counts as a JSON document. The shape
// mirrors the Machine Boss JSON format, with each transition carrying
// an `expected_count` field; this is the JSON you'd use to drive
// downstream tooling (parameter-fitting, visualisation, etc.).
let mut out = String::new();
fb.to_machine_json(&mut out);
std::fs::write("counts.json", &out).unwrap();
```

The emitted JSON looks like

```json
{
  "state": [
    {"n": 0,
     "trans": [
       {"to": 5, "weight": "pNoDescendants[A]", "expected_count": 0.234},
       {"to": 7, "out": "[\"A\",\"\"],\"\"", "weight": {"*": [...]}, "expected_count": 1.018e-3}
     ]},
    …
  ],
  "defs": { … },
  "cons": { … }
}
```

— so it parses with any JSON library, drops in to `boss` (with
`expected_count` as an unrecognized field), and is suitable as a basis
for parameter re-estimation: each transition's expected count is the
sufficient statistic for one EM update of its weight. Bucket granularity
matches what `boss --counts` produces on the same composed machine, so
you can cross-check by piping the machine through `boss --counts -P …`
and comparing the `expected_count` fields transition-for-transition.

### Numerical notes

  - All four helpers return **log** posteriors. Convert with `.exp()`.
    Infeasible transitions (predecessor cell out of range, or observed
    symbols don't match) return `f64::NEG_INFINITY`.
  - The lse cutoff for cell-level posteriors is *not* applied — these
    helpers compute single (cell, transition) terms, not lse'd sums, so
    there's no cutoff to skip. Only `forward_backward_counts` and the
    matrix DPs apply the cutoff.
  - Forward and Backward should agree to ~1e-15 on the same inputs; if
    they diverge, your `lw` is inconsistent (e.g. you ran Forward with
    one `Params` and Backward with another). The default-set test
    `test-rust-codegen-forward-backward` enforces this invariant.

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
  - **lse cutoff** — `log_sum_exp(a, b)` returns `max(a, b)` when
    `|a - b| > 36` nats, since the contribution `log1p(exp(-|a-b|)) ≈
    exp(-36) ≈ 2.3e-16` is smaller than `ulp(hi)` for any cell value
    `hi ≪ -5` (log-probabilities in this DP are deeply negative, so
    `ulp(hi) > 1.78e-15`); the addition is then a no-op. This skips two
    transcendental calls (`exp` and `ln_1p`) for negligible contributions
    and gives Forward another ~16% speedup. (The cutoff would not be
    safe for an algorithm whose accumulators stay near zero — but for
    phylo-composed generators they do not.)

One-time costs for the quartet:
  - codegen:  ~18 s  (boss writes the crate)
  - rustc:    ~36 s  (release build with `lto = true`)

## Verification

  - `make test-rust-codegen-echo` — trivial echo branch transducer; codegen
    Forward matches `boss --phylo-clamp -L` to floating-point precision.
  - `make test-rust-codegen-tkf91` — TKF91 + JC on `(A,B)P;`. Compared
    against an exact-lse Python Forward; observed `|fwd-ref|` ≈ 0 in our
    runs, enforced tolerance 1e-12.
  - `make test-rust-codegen-tkf92-triad` *(manual, not in default test set)* —
    TKF92 + HKY85 on `(A,B,C)X;`. Full Forward reference comparison via
    Python multidim DP. Observed `|fwd-ref|` ≈ 1.8e-15, enforced 1e-12.
  - `make test-rust-codegen-tkf92-quartet` *(manual, not in default test set)* —
    TKF92 + HKY85 on `(A,B,(C,D)Y)X;`. End-to-end smoke test plus the length
    sweep table above.

## Skeleton-bake mode

The `--phylo-skeleton --codegen DIR --rust` invocation emits a **Rust crate
that contains only the inputs to phylo composition**, not the result.
The crate carries faithful ports of the WFST algebra (compose, intersect,
waitingMachine, ergodicMachine, advanceSort, processCycles, plus the
buildSubtree recursion) and runs them at startup to expand the unary
skeleton into the full multidimensional phylo machine.

This trades codegen-time work (which dominates the straight-line mode for
wide alphabets and deep trees) for crate-startup work, and shrinks the
emitted source dramatically: the bake is essentially T + M_skel + Newick,
all alphabet-independent or constant-size.

### What gets emitted

```
$DIR/
├── Cargo.toml          # serde_json dep, lto = true
└── src/
    ├── lib.rs          # 4 baked &'static str constants + prebuild()
    ├── weight_algebra.rs   # JSON WeightExpr evaluator
    ├── machine.rs          # Machine struct + JSON ingest + the WFST algebra
    │                       # (compose, intersect, waiting_machine,
    │                       # ergodic_machine, advance_sort, process_cycles,
    │                       # rename_for_branch, …) — all faithful Rust
    │                       # ports of the corresponding C++.
    ├── phylo.rs            # Newick parser + buildSubtree recursion
    └── forward.rs          # multidim Forward DP over the prebuild()'d machine
```

`lib.rs` exposes:

```rust
pub static T_JSON:       &str;     // canonical branch transducer (un-renamed)
pub static M_SKEL_JSON:  &str;     // unary phylo skeleton (placeholder emits)
pub static TREE_NEWICK:  &str;     // Newick tree string
pub static TIME_PARAM:   &str;     // name of T's per-branch time parameter

/// Build M_full from the baked T + tree by running the Rust port of the
/// WFST algebra. Returns a `machine::Machine` with symbolic
/// `WeightAlgebra` JSON expressions on each transition.
pub fn prebuild() -> machine::Machine;
```

`forward.rs` exposes a single function:

```rust
pub fn forward(m: &Machine, params: &Params, leaves: &[Vec<String>]) -> f64;
```

`Params` is a `HashMap<String, f64>` over the free parameters of T and the
per-branch time parameters that `phylo_intersect` introduces (e.g.
`time[A]`, `time[B]`, …). The Forward DP is a bit-exact mirror of
`t/rust/_phylo_ref.py::multidim_forward` — same exact log_sum_exp via
`(lo - hi).exp().ln_1p()`, same iteration order.

### End-to-end recipe — TKF92 + HKY85 on `(((A,B)P,C)Q)D;`

This is the canonical complex case: a depth-3 caterpillar tree with
**three internal nodes (P, Q, D)**, the TKF92 indel model on each branch,
and an HKY85 substitution kernel. With the straight-line codegen the
emitted `lib.rs` would be hundreds of MB; with the skeleton bake the
crate is small (T + M_skel + tree fit in tens of KB) and the WFST
expansion happens once at startup.

```bash
# 1) Build boss
make

# 2) Bake. We use a TKF92-DNA-HKY85 root + branch transducer (the
#    canonical TKF92 setup) and the (((A,B)P,C)Q)D; topology.
bin/boss \
  --tkf92-root-dna-hky85 \
  -m \
  --begin --tkf92-branch-dna-hky85 \
  --phylo-tree-string '(((A,B)P,C)Q)D;' --phylo-time-param time \
  --end \
  --phylo-skeleton --codegen path/to/depth3 --rust
```

This emits the four baked constants + the Rust modules listed above. The
`M_skel` is alphabet-independent; `T_JSON` is the small (~5-state) TKF92
branch transducer; the tree string is 16 bytes.

```bash
# 3) Compile (release).
cd path/to/depth3
cargo build --release
```

```rust
// 4) Use it — examples/run.rs:
use phylo_skeleton::forward::forward;
use phylo_skeleton::weight_algebra::Params;

fn main() {
    let m = phylo_skeleton::prebuild();    // runs WFST algebra → M_full
    let mut params = Params::new();
    // TKF92 indel parameters
    params.insert("insRate".into(),  0.05);
    params.insert("delRate".into(),  0.06);
    params.insert("r".into(),        0.4);
    // HKY85 substitution parameters
    params.insert("tsRatio".into(),  2.0);
    params.insert("pi_A".into(),     0.30);
    params.insert("pi_C".into(),     0.20);
    params.insert("pi_G".into(),     0.25);
    params.insert("pi_T".into(),     0.25);
    // Per-branch times (one per non-root node — D is the root here)
    params.insert("time[A]".into(),  0.10);
    params.insert("time[B]".into(),  0.20);
    params.insert("time[P]".into(),  0.15);
    params.insert("time[C]".into(),  0.18);
    params.insert("time[Q]".into(),  0.25);

    let leaves: Vec<Vec<String>> = vec![
        "ACGT".chars().map(|c| c.to_string()).collect(),  // A
        "ACG" .chars().map(|c| c.to_string()).collect(),  // B
        "ACT" .chars().map(|c| c.to_string()).collect(),  // C
    ];
    let lk = forward(&m, &params, &leaves);
    println!("forward log-likelihood = {}", lk);
}
```

```bash
cargo run --release --example run
```

### Verification

  - `make test-phylo-skeleton-bake` — bakes TKF91-DNA-JC on `(A,B)P;`
    and `(A,B,C)R;`, asserts the four baked constants parse, asserts the
    Rust ports of `compose` / `intersect` / `waiting_machine` /
    `ergodic_machine` / `advance_sort` / `process_cycles` produce
    state counts matching C++ M_full, and asserts
    `forward(prebuild(), …)` matches the exact-lse Python multidim
    Forward bit-exactly to within 1e-12. Default-set test.
  - `make test-phylo-skeleton-bake-deep` *(manual, not in default test set)* —
    same checks on `(((A,B)P,C)Q)D;` and `((A,B)P,(C,D)Q)R;`. State
    count, state ID set, and forward log-likelihood all match C++
    exactly (271 / 271 IDs match by index for the depth-3 caterpillar).

### Tradeoffs vs. straight-line codegen

|  | Straight-line | Skeleton bake |
|---|---|---|
| Codegen time | Dominated by symbolic weight expansion (seconds–minutes) | Trivial (writes 4 strings + ports) |
| Emitted source size | Scales with M_full (10s–100s of MB on quartets) | Constant-ish (~tens of KB) |
| Rust compile time | Heavy (linear in source size) | Light |
| Crate startup time | Negligible (DP is straight-line) | Runs WFST algebra (seconds–minutes on deep trees, see [perf note](#performance-status)) |
| Forward / cell | Fast (specialised, sharded, lse cutoff applied) | Slower (interpreter over symbolic weights, no sharding yet) |
| Useful for | Hot inner-loop production | Reference correctness, exploratory work, deep trees where straight-line crate doesn't compile |

#### Performance status

The Rust port of the WFST algebra is currently a faithful (un-optimised)
translation of the C++. `prebuild()` on `(((A,B)P,C)Q)D;` with TKF91-DNA-JC
takes ~3 minutes wall-clock — the symbolic weight expressions blow up
through the nested `compose` / `intersect` calls and the per-stage
`ergodic_machine` clones dominate runtime. Known follow-ups:
amortising state cloning across the post-processing chain and
constant-folding weight expressions when free parameters are bound.
The output machine is bit-identical to C++ M_full (state count,
state IDs in order, transition counts, and forward log-likelihood
to 1e-12 floating-point noise).

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
  - **Alphabet scaling**: codegen output size grows with both the number
    of states and the number of distinct emission profiles, so wide
    alphabets (e.g. 20-AA protein with `tkf92-branch-prot-f81`) emit much
    larger crates than DNA. The default
    [Felsenstein-style sub-expression sharing](/phylogeny/#felsenstein-style-sub-expression-sharing)
    on the phylo composition shrinks both `lib.rs` and `machine.json`
    (33% / 57% smaller on a TKF92+F81 protein binary tree, more on
    deeper trees), but the DP grid itself still scales as alphabet × L
    so very wide alphabets remain expensive. The DNA quartet stays
    comfortably within ~40 MB / ~50 s.
