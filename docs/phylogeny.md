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

## Multidimensional Forward via Rust codegen

For a faster (and more numerically precise) Forward / Viterbi over a
phylogenetic composition, see [Rust Codegen](/rust-codegen/). It emits a
self-contained Rust crate that computes a multidimensional DP indexed by
one axis per leaf, with the tree topology and per-branch parameters
specialised at codegen time. Detailed instructions for the TKF92 +
HKY85 quartet on `(A,B,(C,D)Y)X;` are provided there.

## Limitations

- Pair-token decoding back into per-leaf streams is not provided as a
  built-in op; for now, write a downstream "splitter" transducer per
  application. The pair-token nesting carries enough structure to do
  this mechanically.
- No JAX or JS port yet; the JAX (reference) and WebGPU tiers are
  candidates for a future cross-tier consistency story.
