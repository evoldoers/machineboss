---
title: Phylogenetic Intersection
nav_order: 7.5
permalink: /phylogeny/
---

# Phylogenetic Intersection

`--phylo-tree FILE` (or `--phylo-tree-string STR`) is a postfix operator
that takes the top-of-stack machine, treats it as a *branch transducer*
`T`, and builds a phylogenetic intersection over a binary Newick tree.
The result is a transducer whose input alphabet is the symbol set at the
root of the tree, and whose output alphabet is the *pair-token* alphabet
encoding column-wise observations across all leaves (see
[Pair tokens](composition.html#pair-tokens) for the encoding convention).

## Algorithm

For a binary tree, climbing from leaves to root:

```
build(node):
  if node is a leaf:
    return wildEcho(T.outputAlphabet)
  else (node has children u, w):
    T_u = T with timeParam renamed timeParam[<u.name>]
    T_w = T with timeParam renamed timeParam[<w.name>]
    return intersect(compose(T_u, build(u)),
                     compose(T_w, build(w)))
```

So every edge of the tree contributes one copy of `T`, parametrised by
its child node's name; leaves contribute the identity over `T`'s output
alphabet; internal nodes intersect on their shared parent input.

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
recursive intersection: a tree of depth *d* produces output tokens of
shape `{...{leaf_1:leaf_2}:...:leaf_d}`, with the wrapping rule
(introduced by `--intersect`) automatically nesting one extra level of
braces per intersection. For example:

{% raw %}
| Tree | Sample column-token |
|---|---|
| `(A,B)P;` | `0:1` (A=0, B=1) |
| `((A,B)C,D)E;` | `{0:1}:1` (A=0, B=1, D=1) |
| `(((A,B)C,D)E,F)G;` | `{{0:1}:1}:0` (A=0, B=1, D=1, F=0) |
{% endraw %}

Reading a token: outermost `:` separates the root's two subtrees; each
side, if itself a pair, is wrapped in `{...}`. Recursing into the
wrapping reveals the per-leaf symbols in tree order.

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

# Forward log-likelihood: parent="A", joint MSA column = "A:A".
echo '{"sequence":["A"]}'   > parent.json
echo '{"sequence":["A:A"]}' > cols.json
echo '{"delRate":0.02,"insRate":0.01,"time[sibling1]":0.1,"time[sibling2]":0.2}' > params.json

boss --generate-json parent.json -m \
     --begin --preset tkf91-branch-dna-jc \
             --phylo-tree-string '(sibling1,sibling2)parent;' \
             --phylo-time-param time --end \
     --recognize-json cols.json -P params.json -L
```

## Validation

The Newick tree must be strictly binary (every internal node has
exactly two children). If the branch transducer has a parameter named
`timeParam`, every non-root node must have a non-empty name, and node
names must be unique across the tree. Violations raise an error before
the intersection is built.

## Limitations

- Strictly binary trees only. Multifurcations require pre-resolution
  (e.g. by inserting zero-length internal branches).
- Pair-token decoding back into per-leaf streams is not provided as a
  built-in op; for now, write a downstream "splitter" transducer per
  application. The pair-token nesting carries enough structure to do
  this mechanically.
- No JAX or JS port yet; the JAX (reference) and WebGPU tiers are
  candidates for a future cross-tier consistency story.
