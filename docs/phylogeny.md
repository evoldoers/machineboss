---
title: Phylogenetic Intersection
nav_order: 7.5
permalink: /phylogeny/
---

# Phylogenetic Intersection

`--phylo-tree FILE` (or `--phylo-tree-string STR`) is a postfix operator
that takes the top-of-stack machine, treats it as a *branch transducer*
`T`, and builds a phylogenetic intersection over a Newick tree of
arbitrary topology.  The result is a transducer whose input alphabet is
the symbol set at the root of the tree, and whose output alphabet is the
*pair-token* alphabet encoding column-wise observations across all
leaves (see [Pair tokens](composition.html#pair-tokens) for the encoding
convention).

## Algorithm

Climbing from leaves to root:

```
build(node):
  if node is a leaf:
    return wildEcho(T.outputAlphabet)
  if node has one child u:
    T_u = T with timeParam renamed timeParam[<u.name>]
    return compose(T_u, build(u))
  if node has children u_1, u_2, ..., u_n   (n >= 2):
    branch_i = compose(T_{u_i}, build(u_i))   # T_{u_i} is T renamed for u_i
    return intersect(... intersect(branch_1, branch_2) ..., branch_n)
```

So every edge of the tree contributes one copy of `T`, parametrised by
its child node's name; leaves contribute the identity over `T`'s output
alphabet; internal nodes with two or more children intersect their
descendant branches pairwise (polytomies fold-left into iterated
intersection); degree-1 internal nodes pass through.

Per-branch parameter renaming is automatic: every reference to
`timeParam` in a transition weight, every key in `T.funcs.defs` (and
their internal references), and every occurrence in `T.cons` is suffixed
with `[<node.name>]`. Free parameters that are not `timeParam` and not
keys of `defs` are left alone — these are typically global rate
constants like `delRate` and `insRate` and remain shared across all
branches.

If the branch transducer has *no* parameter named `timeParam`, no
renaming is done and node names are not required.

## Output token structure

The pair-token machinery from `--intersect` cascades naturally through
recursive intersection: a tree contributing *d* intersection operations
produces output tokens with up to *d* nested wrappings.  With the
defaults (separator `,`, delimiters `[]`), a tree's column tokens look
like comma-separated, square-bracket-wrapped JSON-ish values:

{% raw %}
| Tree | Sample column-token |
|---|---|
| `(A,B)P;` | `0,1` (A=0, B=1) |
| `(A,B,C)P;` | `[0,1],1` (A=0, B=1, C=1) |
| `((A,B)C,D)E;` | `[0,1],1` (A=0, B=1, D=1) |
| `(((A,B)C,D)E,F)G;` | `[[0,1],1],0` (A=0, B=1, D=1, F=0) |
{% endraw %}

Reading a token: the outermost `,` separates the root's left subtree
from the rest; each side, if itself a pair, is wrapped in `[...]`.
Recursing into the wrappings reveals the per-leaf symbols in tree order.

If you'd rather work with structured JSON than wrapped strings, pass
`--pair-json` to switch the encoder to nested JSON arrays (`[[0,1],1]`
for the depth-2 example above) — see
[JSON pair tokens](composition.html#json-pair-tokens).

## Branch lengths

Newick branch lengths (the `:length` annotations) are extracted into
`timeParam[<node.name>]` parameter values. Pass
`--phylo-params-out FILE` to write a JSON parameter-assignment file
that can be passed back via `-P FILE` for inference.

## CLI

```bash
# Build the TKF91 fork triad over (sibling1, sibling2)parent;
# tkf91-branch-dna-jc's time parameter is named "time" (not the default "t"),
# so we override --phylo-time-param.
boss --preset tkf91-branch-dna-jc \
     --phylo-tree-string '(sibling1:0.1,sibling2:0.2)parent;' \
     --phylo-time-param time \
     --phylo-params-out triad-params.json

# Forward log-likelihood: parent="A", joint MSA column = "A,A".
echo '{"sequence":["A"]}'   > parent.json
echo '{"sequence":["A,A"]}' > cols.json
echo '{"delRate":0.02,"insRate":0.01,"time[sibling1]":0.1,"time[sibling2]":0.2}' > params.json

boss --generate-json parent.json -m \
     --begin --preset tkf91-branch-dna-jc \
             --phylo-tree-string '(sibling1,sibling2)parent;' \
             --phylo-time-param time --end \
     --recognize-json cols.json -P params.json -L
```

## Validation

The Newick tree may be of arbitrary topology: leaves (degree 0),
degree-1 internal nodes, and polytomies (degree ≥ 2) are all accepted.
If the branch transducer has a parameter named `timeParam`, every
non-root node must have a non-empty name, and node names must be unique
across the tree. Violations raise an error before the intersection is
built.

## Felsenstein-style sub-expression sharing

After the recursive intersect/compose has built the phylo machine, every
transition's weight is a complex algebraic expression — typically a sum of
products with one term per parent character, repeated across many
transitions that share the same sub-tree contribution. By default the
phylo build pipes the resulting machine through a **value-based common
sub-expression elimination** pass that hoists shared sub-expressions into
auto-named entries in `funcs.defs` (with names `_phy0`, `_phy1`, ...) and
replaces every occurrence with a parameter reference. This is the same
optimization Felsenstein pruning uses in tree-likelihood DP: gather the
intermediate sum-products once, store them as named intermediates, and
have transitions reference them by name.

Concrete savings on a TKF92 + F81 protein binary tree `(A,B)P;`:

| | Felsenstein on (default) | Felsenstein off |
|---|---|---|
| `machine.json` size | 6.9 MB | 16.0 MB |
| emitted `lib.rs`   | 3.3 MB |  4.9 MB |
| `cargo build --release` | 3.7 s | 4.5 s |

The savings grow with model complexity (alphabet size × tree depth):
the TKF92 protein quartet is the biggest current beneficiary.

To disable, pass `--phylo-no-felsenstein`. The Forward log-likelihood is
identical to floating-point precision in either mode (and bit-exact under
boss's lookup-table log-sum-exp), so the legacy form is only useful for
cross-checking and for inspecting the raw transition-weight expressions.

## Unary skeleton (structural-only fast path)

`--phylo-skeleton` runs the recursive intersect/compose over a unary-
alphabet "skeleton" of the branch transducer. Every emit symbol is replaced
with a single placeholder `*` and every emit weight is set to 1, so the
|Σ|^k column-emission family at each emit-class collapses into a single
placeholder transition during composition. The resulting machine has the
same accessible state set as the full phylo machine (same topology) and
the same per-leaf emit/silent pattern in its pair-tokens, but carries no
per-symbol substitution weights.

Concrete savings on a TKF92 + F81 protein binary tree `(A,B)P;` with
`--phylo-no-felsenstein`:

| | Full | Skeleton | Speedup |
|---|---|---|---|
| Wall time           | 1.7 s   | 0.03 s | ~57× |
| `machine.json` size | 16 MB   | 8.8 KB | ~1800× |
| Transitions         | 106 340 | 92     | ~1156× |
| States              | 25      | 25     | (identical topology) |

On the protein quartet `((A,B)P,(C,D)Q)R;` the full path requires gigabytes
of intermediate machine; the skeleton finishes in 0.5 s producing a 2.4 MB
machine with ~3.9 K states.

The skeleton path is currently intended as a fast topology-only pass (e.g.
to inspect state-set growth, to sanity-check pair-token shapes for new
trees, or to drive future per-column symbol expansion). It is not a drop-in
replacement for inference: you cannot compute a Forward log-likelihood from
the skeleton output directly, since the per-symbol weights are not present.

### Skeleton state names and decoding

After `--phylo-skeleton`, every M_skel state's `id` is an ordered nested
JSON array that recursively mirrors the tree topology (the
[composite-state-name](composition.html#composite-state-names) convention
applies here). For a binary tree `(A,B)P;` the `id` is
`[[T_BA_state, ['*']], [T_BB_state, ['*']]]` — the outer pair encodes the
intersect of the two children at the root, each inner pair encodes a
compose of `T` (renamed for that branch) with the leaf's wildEcho. For
deeper trees the structure recurses one level per intersect/compose, with
`{"wait": ...}` wraps appearing inside whenever `waitingMachine` introduced
wait-states. A consumer can recover the per-branch T state at every M_skel
state by walking this name in lockstep with the tree, unwrapping any
`{"wait": ...}` it encounters. See
`t/check_phylo_skeleton_expand.py` for a reference Python decoder.

### Symbol expansion (validation tool)

`t/check_phylo_skeleton_expand.py` is a Python validator that, given M_skel
plus the original branch transducer and the tree, expands each emit
transition back into the per-symbol family by:

1. Decoding M_skel's state names into per-branch T-states (per the
   convention above).
2. For each emit transition, looking up T's symbol-aware transitions on
   each advancing branch and cross-multiplying under the intersect
   input-sync constraint to produce expanded `(in, root_sym; out,
   pair_token; weight)` transitions.
3. Copying silent transitions verbatim (M_skel preserves their full
   structural weights, including any chain-collapsed factors from
   ergodicMachine, since skeletonisation only resets *emit* weights to 1).
4. Feeding the expanded machine through `boss -L` and comparing the Forward
   log-likelihood bit-exactly against the legacy `--phylo-no-felsenstein`
   path on the same observation.

The validator currently passes on trees with no internal nodes (trees in
which every leaf is a direct child of the root, e.g. `(A,B)P;` or
`(A,B,C)P;`). Trees with internal nodes (e.g. the protein quartet
`((A,B)P,(C,D)Q)R;`) require an additional Felsenstein-per-column sum over
internal-node symbol assignments — every such weight in the legacy machine
factors as `Σ_internal_symbol Π_branch P_branch[parent_sym, child_sym]`,
which the cross-product expansion cannot reproduce on its own. Generalising
the validator is straightforward (post-order Felsenstein recursion at each
internal node) and is intended as the next increment.

A follow-up will port the expansion to C++ behind a CLI flag once the
Felsenstein-per-column generalisation is in place.

## Multidimensional Forward via Rust codegen

For a faster (and more numerically precise) Forward / Viterbi over a
phylogenetic composition, see [Rust Codegen](/rust-codegen/). It emits a
self-contained Rust crate that computes a multidimensional DP indexed by
one axis per leaf, with the tree topology and per-branch parameters
specialised at codegen time. Detailed instructions for the TKF92 +
HKY85 quartet on `(A,B,(C,D)Y)X;` are provided there.

The Rust crate also exposes the **Backward** matrix and a small set of
helpers for **posterior probabilities** — state posteriors at any
alignment cell, transition posteriors at any cell (silent or emitting),
and Forward-Backward expected transition counts — plus a JSON helper
that emits a Machine Boss-shaped JSON document with the counts attached
per transition. See [Posterior probabilities](/rust-codegen/#posterior-probabilities)
in the Rust codegen page for the math, the API, and worked examples.

## Limitations

- Pair-token decoding back into per-leaf streams is not provided as a
  built-in op; for now, write a downstream "splitter" transducer per
  application. The pair-token nesting carries enough structure to do
  this mechanically.
- No JAX or JS port yet; the JAX (reference) and WebGPU tiers are
  candidates for a future cross-tier consistency story.
