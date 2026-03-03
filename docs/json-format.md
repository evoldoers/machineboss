---
title: JSON Format
nav_order: 3
permalink: /json-format/
---

# Machine Boss JSON Format Reference

Machine Boss uses a JSON representation of
[weighted finite-state transducers](https://en.wikipedia.org/wiki/Finite-state_transducer) (WFSTs).
This page documents every schema the program defines, with examples.
The authoritative schemas live in the [`schema/`](https://github.com/evoldoers/machineboss/tree/master/schema) directory of the repository.

{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Transducer (`machine.json`)

A transducer is defined as a tuple *(Φ, Σ, Γ, ω)* where
*Φ* is the state space (including start and end states),
*Σ* is the input alphabet,
*Γ* is the output alphabet, and
*ω(α, β, σ, γ)* is the transition weight.

In the JSON format the **start state is always the first** element of the `state` array and the **end state is always the last**.
A single-state machine where the sole state is both start and end is allowed.

### Concrete transducer

The most common form is a concrete state array:

```json
{
  "state": [
    {
      "id": "S",
      "trans": [
        { "in": "0", "out": "0", "to": "S", "weight": "p" },
        { "in": "0", "out": "1", "to": "S", "weight": "q" },
        { "in": "1", "out": "1", "to": "S", "weight": "p" },
        { "in": "1", "out": "0", "to": "S", "weight": "q" }
      ]
    }
  ]
}
```

This is the [binary symmetric channel](https://en.wikipedia.org/wiki/Binary_symmetric_channel): a single-state transducer that copies each bit with probability `p` and flips it with probability `q`.

### State object

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | any (not number) | id or n | State identifier (string, object, array, boolean, or null). Used in `"to"` references. |
| `n` | number | id or n | Numeric state index (alternative to `id`). |
| `trans` | array of transitions | no | Outgoing transitions. Omitted for the end state (or any absorbing state). |

At least one of `id` or `n` must be present. The `id` field must not be a number (use `n` for numeric state indices).

### Transition object

Each transition can take one of three forms:

| Field | Type | Required | Description |
|---|---|---|---|
| `to` | any | yes | Destination state (matches a state's `id` or `n`). |
| `in` | string | no | Input symbol consumed. Omitted for silent (epsilon) transitions on the input tape. |
| `out` | string | no | Output symbol emitted. Omitted for silent transitions on the output tape. |
| `weight` | [expr](#weight-expressions-exprjson) | no* | Weight expression. See below. If omitted, weight is 1. |
| `expr` | string | no* | Inline [weight expression string](/expressions/) (alternative to `weight`). |

*A transition must have `weight`, `expr`, or neither (implicit weight 1). These are mutually exclusive.

### Top-level optional fields

| Field | Type | Description |
|---|---|---|
| `state` | array | The state array (required for concrete machines). |
| `defs` | [defs](#parameter--function-definitions-defsjson) | Named function/parameter definitions, referenced in weight expressions. |
| `cons` | [constraints](#normalization-constraints-constraintsjson) | Normalization constraints for model fitting. |
| `params` | array of strings | Explicit list of parameter names used in the machine. |

### Machine expressions (algebraic constructors)

Instead of a concrete `state` array, a machine JSON object can be one of the following algebraic operators applied to sub-machines:

| Operator | Arity | Description |
|---|---|---|
| `compose` | 2 | Transducer composition (matrix multiplication). Break silent cycles. |
| `compose-sum` | 2 | Composition with silent-cycle summation. |
| `compose-unsort` | 2 | Composition without sorting. |
| `concat` | 2 | Concatenation of two machines. |
| `intersect` | 2 | Intersection (pointwise product). Break silent cycles. |
| `intersect-sum` | 2 | Intersection with silent-cycle summation. |
| `intersect-unsort` | 2 | Intersection without sorting. |
| `union` | 2 | Union (pointwise sum). |
| `loop` | 2 | Loop: `x(yx)*`. |
| `opt` | 1 | Zero or one (union with null). |
| `star` | 1 | Kleene star (zero or more). |
| `plus` | 1 | Kleene plus (one or more). |
| `eliminate` | 1 | Eliminate silent transitions. |
| `reverse` | 1 | Reverse the machine. |
| `revcomp` | 1 | Reverse complement. |
| `transpose` | 1 | Swap input and output labels. |

Binary operators take a 2-element array; unary operators take a single machine object.

```json
{
  "compose": [
    { "state": [ {"id":"S", "trans":[{"in":"A","out":"a","to":"S"}]} ] },
    { "state": [ {"id":"S", "trans":[{"in":"a","out":"1","to":"S"}]} ] }
  ]
}
```

## Weight Expressions (`expr.json`)

Transition weights can be constants, parameter references, or algebraic combinations thereof.
A weight expression is one of:

| Form | JSON type / key | Example | Description |
|---|---|---|---|
| Boolean constant | `boolean` | `true` | true = 1, false = 0 |
| Numeric constant | `number` | `0.99` | Literal numeric weight. |
| Parameter reference | `string` | `"p"` | Named parameter, resolved at evaluation time. |
| Inline expression | `{"expr": "..."}` | `{"expr":"p*q"}` | Parsed expression string. |
| Logarithm | `{"log": <expr>}` | `{"log":"p"}` | Natural log of the argument. |
| Exponential | `{"exp": <expr>}` | `{"exp":2}` | Exponential of the argument. |
| Geometric sum | `{"geomsum": <expr>}` | `{"geomsum":"r"}` | Geometric series sum: 1/(1-x). |
| Logical NOT | `{"not": <expr>}` | `{"not":"p"}` | Boolean negation (1-x). |
| Multiply | `{"*": [a, b]}` | `{"*":["p",0.5]}` | Product of two expressions. |
| Add | `{"+": [a, b]}` | `{"+":["p","q"]}` | Sum of two expressions. |
| Divide | `{"/": [a, b]}` | `{"/": [1, "n"]}` | Division. |
| Subtract | `{"-": [a, b]}` | `{"-": [1, "p"]}` | Subtraction. |
| Power | `{"pow": [a, b]}` | `{"pow":["x",2]}` | Exponentiation: a^b. |

All binary operators (`*`, `+`, `/`, `-`, `pow`) take exactly two arguments in an array.
Expressions are recursive: arguments can be any expression form.

### Example: compound weight

```json
{
  "weight": {"*": ["p", {"-": [1, "q"]}]}
}
```

This represents the weight `p × (1 - q)`.

## Parameter Assignments (`params.json`)

A parameter assignment file maps parameter names to numeric values:

```json
{"p": 0.99, "q": 0.01}
```

| Field | Type | Description |
|---|---|---|
| *(any key)* | number | Value assigned to the named parameter. |

Used with the `-P/--params` CLI option to supply parameter values for evaluation, inference, and training.

## Parameter / Function Definitions (`defs.json`)

A definitions file maps parameter names to [weight expressions](#weight-expressions-exprjson), allowing parameters to be defined in terms of other parameters:

```json
{"e": {"*": ["p", "q"]}}
```

| Field | Type | Description |
|---|---|---|
| *(any key)* | [expr](#weight-expressions-exprjson) | Expression defining the named parameter. |

Used with `-F/--functions` or in the `defs` field of a machine file.

## Normalization Constraints (`constraints.json`)

Constraints specify normalization conditions used during Baum-Welch training.

```json
{
  "norm": [["p", "q"], ["a", "b", "c"]],
  "prob": ["x"],
  "rate": ["r"]
}
```

| Field | Type | Description |
|---|---|---|
| `norm` | array of string arrays | Each inner array is a group of parameters that must sum to 1. |
| `prob` | array of strings | Parameters constrained to [0, 1]. |
| `rate` | array of strings | Parameters constrained to be non-negative rates. |

In the example above, the constraints are `p + q = 1` and `a + b + c = 1`.

## Named Sequences (`namedsequence.json`)

A named sequence pairs an optional name with an array of single-character symbols:

```json
{"name": "AGC", "sequence": ["A", "G", "C"]}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `sequence` | array of strings | yes | The symbol sequence (each element is typically a single character). |
| `name` | string | no | Optional descriptive name. |

## Sequence Pairs (`seqpair.json`)

A sequence pair specifies input and output data for inference algorithms. Two forms are supported.

### Sequence form

```json
{
  "input":  {"name": "001", "sequence": ["0","0","1"]},
  "output": {"name": "101", "sequence": ["1","0","1"]}
}
```

### Alignment form

```json
{
  "input":  {"name": "seq1"},
  "output": {"name": "seq2"},
  "alignment": [["A","A"], ["G",""], ["","T"], ["C","C"]]
}
```

In the alignment form, each element is a 2-element array `[input_symbol, output_symbol]`.
Empty strings represent gaps (insertions or deletions).

| Field | Type | Required | Description |
|---|---|---|---|
| `input` | [named sequence](#named-sequences-namedsequencejson) | yes | The input sequence. |
| `output` | [named sequence](#named-sequences-namedsequencejson) | yes | The output sequence. |
| `alignment` | array of [string,string] | alt. | Explicit alignment path (alternative to full sequences). |
| `meta` | any | no | Optional metadata. |

## Sequence-Pair Lists (`seqpairlist.json`)

A sequence-pair list is simply a JSON array of [sequence pair](#sequence-pairs-seqpairjson) objects, used for batch training and alignment:

```json
[
  {"input":{"name":"001","sequence":["0","0","1"]},
   "output":{"name":"101","sequence":["1","0","1"]}},
  {"input":{"name":"01","sequence":["0","1"]},
   "output":{"name":"10","sequence":["1","0"]}}
]
```

Used with the `-D/--data` CLI option.

## Schema Files

The JSON Schema source files are located in the [`schema/`](https://github.com/evoldoers/machineboss/tree/master/schema) directory of the repository:

- [`schema/machine.json`](https://github.com/evoldoers/machineboss/blob/master/schema/machine.json) --- Transducer schema
- [`schema/expr.json`](https://github.com/evoldoers/machineboss/blob/master/schema/expr.json) --- Weight expression schema
- [`schema/params.json`](https://github.com/evoldoers/machineboss/blob/master/schema/params.json) --- Parameter assignments schema
- [`schema/defs.json`](https://github.com/evoldoers/machineboss/blob/master/schema/defs.json) --- Function definitions schema
- [`schema/constraints.json`](https://github.com/evoldoers/machineboss/blob/master/schema/constraints.json) --- Normalization constraints schema
- [`schema/namedsequence.json`](https://github.com/evoldoers/machineboss/blob/master/schema/namedsequence.json) --- Named sequence schema
- [`schema/seqpair.json`](https://github.com/evoldoers/machineboss/blob/master/schema/seqpair.json) --- Sequence-pair schema
- [`schema/seqpairlist.json`](https://github.com/evoldoers/machineboss/blob/master/schema/seqpairlist.json) --- Sequence-pair list schema
