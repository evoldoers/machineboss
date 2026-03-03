---
title: Expression Language
nav_order: 4
permalink: /expressions/
---

# Machine Boss Weight Expression Language

Machine Boss includes a small expression language for specifying transition weights.
Expressions can appear in three places:

1. On the command line, via the `--weight` option (shorthand `#`)
2. In JSON machine files, via the `"expr"` field on transitions (as an alternative to the structured `"weight"` field)
3. In the `--weight-input` and `--weight-output` options (with macro expansion; see [Macros](#macros--and-) below)

The expression language is a conventional infix arithmetic notation with named parameters.
It is parsed by a PEG grammar (defined in `src/grammars/expr.h`) and compiled into the same
internal weight-expression tree used by the [JSON weight expression format](/json-format/#weight-expressions-exprjson).

{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Formal Grammar

The expression parser is defined by the following PEG grammar
(from `src/grammars/expr.h`):

```
Term       <- _ Factor (Add / Sub)*
Add        <- '+' _ Factor
Sub        <- '-' _ Factor
Factor     <- Power (Mul / Div)*
Mul        <- '*' _ Power
Div        <- '/' _ Power
Power      <- Primary '^' _ Primary / Primary
Primary    <- (Parens / Function / NegateProb / Negative / Number / Variable) _
Parens     <- '(' Term ')'
Function   <- Exp / Log
Exp        <- 'exp' _ '(' _ Term _ ')' / 'e' _ '^' _ Primary
Log        <- 'log' _ '(' _ Term _ ')'
NegateProb <- '!' _ Primary
Negative   <- '-' _ Primary
Number     <- Sign _ (Scientific / IntOrFloat)
IntOrFloat <- Float / Integer
Integer    <- [0-9]+
Float      <- Integer '.' Integer / '.' Integer
Scientific <- IntOrFloat ('e'/'E') Sign Integer
Variable   <- '$' identifier
identifier <- [a-zA-Z] [a-zA-Z0-9]*
_          <- [ \t\r\n]*
```

## Value Types

| Type | Syntax | Examples |
|---|---|---|
| Integer | `[0-9]+` | `0`, `42`, `100` |
| Float | `INT.INT` or `.INT` | `0.99`, `3.14`, `.5` |
| Scientific | `FLOAT e SIGN INT` | `1e-3`, `2.5E10`, `.1e+2` |
| Variable (parameter) | `$name` | `$p`, `$rate`, `$x1` |

Numbers may have an optional leading sign (`+` or `-`).
Variable names begin with a letter and may contain letters and digits.

## Operators and Precedence

Operators are listed from lowest to highest precedence:

| Precedence | Operator | Syntax | Description | Associativity |
|---|---|---|---|---|
| 1 (lowest) | Addition | `a + b` | Sum | Left |
| 1 | Subtraction | `a - b` | Difference | Left |
| 2 | Multiplication | `a * b` | Product | Left |
| 2 | Division | `a / b` | Quotient | Left |
| 3 (highest) | Exponentiation | `a ^ b` | Power (a^b) | Right |

Parentheses `( )` override precedence as usual.

## Functions

| Function | Syntax | Description |
|---|---|---|
| Exponential | `exp(x)` or `e^x` | Natural exponential, e^x |
| Logarithm | `log(x)` | Natural logarithm, ln(x) |
| Negate probability | `!x` | Probability complement: 1 - x |
| Unary minus | `-x` | Arithmetic negation: 0 - x |

The `!` operator is particularly useful for probability models: `!$p` computes `1 - $p`,
so if `$p` is the probability of one outcome, `!$p` is the probability of the complementary outcome
without requiring a separate parameter.

## Variables (Parameters)

Variables are named parameters, written with a `$` prefix: `$p`, `$rate`, `$x1`.
A variable name must start with a letter and may contain letters and digits.

Parameter values can be supplied via:

- `--params FILE` --- a JSON file mapping parameter names to numeric values (e.g. `{"p": 0.99, "q": 0.01}`)
- `--functions FILE` --- a JSON file mapping parameter names to expressions
- `--use-defaults` --- use uniform/unit defaults for unbound parameters
- The `defs` field within a machine JSON file

When a parameter appears in an expression but is not bound, it remains symbolic.
The `--show-params` option lists all unbound parameters in the final machine.

## Macros (`%` and `#`)

The `--weight-input` and `--weight-output` options accept expression strings
with two macro placeholders that are expanded before parsing:

| Macro | Expands to | Example |
|---|---|---|
| `%` | The current input/output symbol name | `$p%` becomes `$pA`, `$pC`, `$pG`, `$pT` for a DNA alphabet |
| `#` | The alphabet size (as a number) | `1/#` becomes `1/4` for a DNA alphabet |

These macros are used internally by several shorthand options:

- `--generate-iid` and `--recognize-iid` use the default macro `$p%`,
  creating a per-symbol parameter (e.g. `$pA`, `$pC`, `$pG`, `$pT`)
- `--generate-uniform` and `--recognize-uniform` use `1/#`,
  giving each symbol weight 1/(alphabet size)

### Example

```bash
# Weight each DNA output by a per-nucleotide parameter:
boss --generate-wild-dna --weight-output '$p%'

# Weight each DNA input uniformly (1/4):
boss --recognize-wild-dna --weight-input '1/#'
```

## Equivalence with JSON Weight Format

Every expression string is compiled into the same internal representation
as the [JSON weight expression format](/json-format/#weight-expressions-exprjson).
The two notations are interchangeable:

| Expression string | Equivalent JSON weight |
|---|---|
| `$p` | `"p"` |
| `$p*$q` | `{"*": ["p", "q"]}` |
| `$p+$q` | `{"+": ["p", "q"]}` |
| `1-$p` | `{"-": [1, "p"]}` |
| `1/$n` | `{"/": [1, "n"]}` |
| `$x^2` | `{"pow": ["x", 2]}` |
| `exp($r)` | `{"exp": "r"}` |
| `log($p)` | `{"log": "p"}` |
| `!$p` | `{"not": "p"}` |
| `0.99` | `0.99` |

In a JSON transition, you can use either notation:

```json
// Using the structured "weight" field (JSON expression tree):
{"to": "S", "in": "0", "out": "0", "weight": {"*": ["p", "q"]}}

// Using the "expr" field (expression string):
{"to": "S", "in": "0", "out": "0", "expr": "$p*$q"}
```

The two forms are mutually exclusive on a single transition.
If neither `"weight"` nor `"expr"` is present, the transition weight defaults to 1.

## Command-Line Examples

### Weighted null transition

```bash
# A single silent transition with weight $p:
boss --weight '$p'

# Shorthand using the '#' opcode:
boss '#' '$p'

# Numeric weight:
boss --weight 0.5

# Compound expression:
boss --weight '$p*$q'
boss --weight '1/2'
boss --weight '$p+$q'
```

### Parameterized models

```bash
# Build a noisy channel with expressions, evaluated with specific parameters:
boss --weight '$p' --evaluate --params '{"p": 0.95}'

# Reciprocal of per-symbol weights:
boss --recognize-wild ACGT --weight-input '$p%' --reciprocal
```

### Using expressions in JSON files

```bash
# bitnoise.json uses expression strings:
# {"to":"S", "in":"0", "out":"0", "weight":"p"}
# Here "p" is a bare parameter name (no $) because it's a JSON weight, not an expr string.
#
# The same machine could use the expr field instead:
# {"to":"S", "in":"0", "out":"0", "expr":"$p"}
```
