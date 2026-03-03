---
title: JSON Output Formats
nav_order: 5
permalink: /json-output/
---

# Machine Boss JSON Output Formats

Machine Boss produces several different JSON output formats depending on the operation.
This document describes the structure of each output format.

{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Machine JSON (default / `--save`)

The default output is the transducer itself in JSON format.
This is the same format used for input; see the [JSON Format Reference](/json-format/) for full details.

```json
{"state":
 [{"n":0,
   "id":"S",
   "trans":[{"to":0,"in":"0","out":"0","weight":"p"},
            {"to":0,"in":"0","out":"1","weight":"q"},
            {"to":0,"in":"1","out":"0","weight":"q"},
            {"to":0,"in":"1","out":"1","weight":"p"}]},
  {"n":1,
   "id":"E"}
 ]
}
```

The start state is always the first element (index 0) and the end state is always last.
Transitions reference destination states by index (`"to"`).
Weights may be numbers, parameter names (strings), or structured expressions.

## Evaluated Machine (`--evaluate`)

The `--evaluate` option outputs the machine with all symbolic weight expressions
replaced by their numeric values (as log-weights). Requires parameters to be supplied
via `--params` or `--use-defaults`.

```bash
boss bitnoise.json -P params.json --evaluate
```

```json
{"state":
 [{"n":0,
   "id":"S",
   "trans":[{"to":0,"in":"0","out":"0","weight":0.99},
            {"to":0,"in":"0","out":"1","weight":0.01},
            {"to":0,"in":"1","out":"0","weight":0.01},
            {"to":0,"in":"1","out":"1","weight":0.99}]},
  {"n":1,
   "id":"E"}
 ]
}
```

All `"weight"` fields become numeric values.
This is useful for inspecting the concrete transition probabilities of a parameterized model.

## Parameters (`--train`)

The `--train` (or `-T`) option runs Baum-Welch EM training and outputs the fitted parameters
as a JSON object mapping parameter names to numeric values:

```json
{"p": 0.985, "q": 0.015}
```

This output can be fed back as input via `--params` for subsequent operations.

## Log-likelihood (`--loglike`)

The `--loglike` (or `-L`) option computes the Forward log-likelihood for each
input/output sequence pair. Output is a JSON array of triples:

```json
[["input_name", "output_name", -1.234],
 ["input2", "output2", -5.678]]
```

Each element is `[input_sequence_name, output_sequence_name, log_likelihood]`.
The log-likelihood is the natural logarithm of the total probability summed over all paths.

## Viterbi (`--viterbi`)

The `--viterbi` (or `-V`) option computes the Viterbi (maximum-likelihood path) log-probability.
The output format is the same as [log-likelihood](#log-likelihood---loglike):

```json
[["input_name", "output_name", -0.987]]
```

The value is the log-probability of the single most-probable path, rather than
the sum over all paths.

## Alignment (`--align`)

The `--align` (or `-A`) option computes the Viterbi alignment.
Output is a [SeqPairList](#seqpairlist-format) with alignment path information:

```json
[{"input":{"name":"101","sequence":["1","0","1"]},
  "output":{"name":"001","sequence":["0","0","1"]},
  "path":{"start":0,
          "trans":[{"to":0,"in":"1","out":"0"},
                   {"to":0,"in":"0","out":"0"},
                   {"to":0,"in":"1","out":"1"}]}}]
```

The `"path"` field contains the Viterbi traceback: a start state and an array
of transitions taken. Each transition includes the destination state and any
input/output symbols consumed/emitted.

## Counts (`--counts`)

The `--counts` (or `-C`) option computes expected transition counts via the
Forward-Backward algorithm. Output is a JSON object mapping parameter names to
their expected counts (partial derivatives of the log-likelihood with respect to
log-parameters):

```json
{"p": 2.95, "q": 0.05}
```

## Encoding

Encoding operations find the most likely output sequence given an input sequence.
All encoding options output a [SeqPairList](#seqpairlist-format):

| Option | Algorithm |
|---|---|
| `--beam-encode` / `-Y` | Beam search |
| `--prefix-encode` | CTC prefix search |
| `--viterbi-encode` | Viterbi traceback |
| `--random-encode` | Stochastic prefix search (use `--seed` for reproducibility) |

```json
[{"input":{"name":"101","sequence":["1","0","1"]},
  "output":{"name":"output","sequence":["1","0","1"]}}]
```

## Decoding

Decoding operations find the most likely input sequence given an output sequence.
All decoding options output a [SeqPairList](#seqpairlist-format):

| Option | Algorithm |
|---|---|
| `--beam-decode` / `-Z` | Beam search |
| `--prefix-decode` | CTC prefix search |
| `--viterbi-decode` | Viterbi traceback |
| `--cool-decode` | Simulated annealing (use `--seed` for reproducibility) |
| `--mcmc-decode` | MCMC search (use `--seed` for reproducibility) |

```json
[{"input":{"name":"input","sequence":["1","0","1"]},
  "output":{"name":"101","sequence":["1","0","1"]}}]
```

## SeqPairList Format

Many operations produce output in *SeqPairList* format: a JSON array of sequence-pair objects.
Each element has `"input"` and `"output"` fields, each containing a named sequence:

```json
[
  {
    "input":  {"name": "seq1", "sequence": ["A", "C", "G"]},
    "output": {"name": "seq2", "sequence": ["A", "G", "G"]}
  }
]
```

Alignment results additionally include a `"path"` field with the Viterbi traceback
(see [Alignment](#alignment---align)).

The sequence `"name"` field typically comes from the input data (e.g. FASTA header
or JSON sequence name). When sequences are specified on the command line
(e.g. `--input-chars`), the name is the character string itself.

## Constraints

Constraint files (loaded via `--constraints`) define normalization constraints
for parameter fitting. The format is a JSON object with constraint groups:

```json
{
  "norm": [["p", "q"]],
  "prob": ["p", "q"],
  "rate": ["r"]
}
```

| Key | Constraint |
|---|---|
| `"norm"` | Groups of parameters that must sum to 1 |
| `"prob"` | Individual parameters constrained to [0, 1] |
| `"rate"` | Parameters constrained to be non-negative |
