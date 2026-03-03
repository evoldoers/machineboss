---
title: Program Reference
nav_order: 2
permalink: /machineboss/
---

# Machine Boss Program Reference
{: .no_toc }

Machine Boss is a command-line toolkit for constructing, manipulating, and applying
[weighted finite-state transducers](https://en.wikipedia.org/wiki/Finite-state_transducer)
in bioinformatics. It supports regular expressions, HMMER profile HMMs, CSV profiles, FASTA sequences,
and a native [JSON transducer format](/json-format/).

Machine Boss is multilingual: algorithms are available in C++, Python/JAX, JavaScript, and WGSL (WebGPU).
The [Python/JAX package](/python/) provides differentiable inference and training algorithms,
supporting HMM decoding heads and neural transducers for integration with deep learning frameworks.
The [WebGPU/JavaScript library](/webgpu/) enables GPU-accelerated inference
directly in the browser, with a pure JavaScript CPU fallback for environments without WebGPU support.

**Citation:** Silvestre-Ryan, Wang, Sharma, Lin, Shen, Dider, and Holmes.
[*Machine Boss: Rapid Prototyping of Bioinformatic Automata.*](https://pubmed.ncbi.nlm.nih.gov/32683444/)
Bioinformatics (2020).

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

### From source (macOS)

```bash
brew install gsl boost htslib pkgconfig
make
# optionally:
npm install    # needed for tests
make test
```

## Overview

The `boss` command builds a transducer from a series of machine expressions given as
command-line arguments, then optionally runs an inference algorithm on data. If no inference
operation is requested, the resulting machine is printed to standard output as JSON
(or as a GraphViz dot file with `--graphviz`).

The general workflow is:

1. Construct or load one or more machines (generators, recognizers, echo machines, presets, files, regex, HMMER models).
2. Combine them using operators (compose, concatenate, intersect, union, Kleene star, etc.).
3. Optionally apply inference (Forward, Viterbi, Baum-Welch, beam search, etc.) with data.

## Constructing Machines

### Generators

Generators produce output sequences (analogous to row vectors).

| Option | Description |
|---|---|
| `--generate-one ACGT` | Generator for any *one* of the specified characters. Like a regex character class. |
| `--generate-wild ACGT` | Generator for any *string* over the given alphabet. Like a regex wildcard `.*`. |
| `--generate-iid ACGT` | Like `--generate-wild` but with parameterized character frequencies. |
| `--generate-uniform ACGT` | Like `--generate-iid` but with uniform weights (1/alphabet size). |
| `--generate-chars AGATTC` | Generator for the single specified string. |
| `--generate-fasta FILE` | Generator for a sequence read from a FASTA file. |
| `--generate-csv FILE` | Generator from a CSV position weight matrix. |
| `--generate-json FILE` | Generator for a sequence from a JSON file. |

Alphabet shortcuts: append `-dna` (`ACGT`), `-rna` (`ACGU`), or `-aa` (20 amino acids)
to any of the above, e.g. `--generate-uniform-dna`.

### Recognizers

Recognizers accept input sequences (analogous to column vectors).
Replace `--generate` with `--recognize` in any of the above.

### Echo (identity) machines

Replace `--generate` with `--echo` for identity machines that copy input to output.

### Regular expressions

| Option | Description |
|---|---|
| `--regex REGEX` | Text recognizer from a regular expression. Use `^` and `$` anchors for global matching. |
| `--dna-regex REGEX` | DNA regex (wildcard `.` matches `ACGT`). |
| `--rna-regex REGEX` | RNA regex (wildcard `.` matches `ACGU`). |
| `--aa-regex REGEX` | Protein regex (wildcard `.` matches 20 amino acids). |

### External formats

| Option | Description |
|---|---|
| `--hmmer FILE.hmm` | Generator from an HMMER3 profile HMM (local alignment mode). |
| `--hmmer-global FILE.hmm` | Generator from an HMMER3 profile HMM (global alignment mode). |
| `--hmmer-plan7 FILE.hmm` | Plan7 generator from an HMMER3 model (single-hit with N/C flanks). |
| `--hmmer-multihit FILE.hmm` | Plan7 generator from an HMMER3 model (multi-hit with J loop). |
| `--jphmm FILE.fa` | Jumping profile HMM generator from a FASTA multiple alignment. |

A bare filename argument (not matching any option) is loaded as a JSON machine file.

## Preset Machines

Selected via `--preset NAME` or `-p NAME`.

| Name | Description |
|---|---|
| `null` | Identity for the empty string. |
| `compdna` | DNA complement (does not reverse). |
| `comprna` | RNA complement (does not reverse). |
| `translate` | Inputs amino acids, outputs codons (reverse translation). |
| `prot2dna` | GeneWise-style model: find a protein in DNA. |
| `psw2dna` | GeneWise-style model with substitutions & indels. |
| `dnapsw` | Probabilistic Smith-Waterman for DNA. |
| `dnapswnbr` | Probabilistic Smith-Waterman for DNA (no between-region model). |
| `protpsw` | Probabilistic Smith-Waterman for protein. |
| `jukescantor` | Jukes-Cantor (1969) DNA substitution model. |
| `tkf91branch` | Thorne-Kishino-Felsenstein (1991) DNA indel model + Jukes-Cantor substitutions. |
| `tkf91root` | Equilibrium distribution of the TKF91 model. |
| `bintern` | Binary (base-2) to ternary (base-3) converter. |
| `terndna` | Ternary to non-repeating DNA. With `bintern`, implements the Goldman *et al.* DNA storage code. |
| `tolower` | Convert text to lower case. |
| `toupper` | Convert text to upper case. |
| `hamming31` | Hamming (3,1) error correction code. |
| `hamming74` | Hamming (7,4) error correction code. |
| `iupacdna` | IUPAC DNA ambiguity codes. |
| `iupacaa` | IUPAC amino acid ambiguity codes. |
| `dna2rna` | Convert DNA (T) to RNA (U). |
| `rna2dna` | Convert RNA (U) to DNA (T). |

## Postfix Operators (Single-Machine Transforms)

| Option | Opcode | Description |
|---|---|---|
| `--zero-or-one` | `?` | Union with null (optional). Like regex `?`. |
| `--kleene-star` | `*` | Zero or more repetitions. Like regex `*`. |
| `--kleene-plus` | `+` | One or more repetitions. Like regex `+`. |
| `--count-copies x` | | Like Kleene star with a dummy counting parameter. |
| `--repeat N` | | Repeat exactly N times. |
| `--reverse` | | Reverse the machine. |
| `--revcomp` | `~` | Reverse complement. |
| `--double-strand` | | Union of machine with its reverse complement. |
| `--transpose` | | Swap input and output labels. |
| `--reciprocal` | | Invert all weight expressions. |
| `--joint-norm` | | Normalize outgoing weights to sum to 1. |
| `--cond-norm` | | Normalize conditionally (per input symbol). |
| `--sort` | | Topological sort, eliminating silent cycles. |
| `--sort-fast` | | Topological sort, breaking silent cycles (faster, destructive). |
| `--sort-cyclic` | | Topological sort preserving silent cycles. |
| `--decode-sort` | | Topological sort of non-outputting transition graph. |
| `--encode-sort` | | Topological sort of non-inputting transition graph. |
| `--full-sort` | | Topological sort of entire transition graph. |
| `--eliminate` | | Eliminate all silent transitions. |
| `--eliminate-states` | | Eliminate states with only silent in/out transitions. |
| `--silence-input` | | Clear input labels (machine becomes a generator). |
| `--silence-output` | | Clear output labels (machine becomes a recognizer). |
| `--copy-output-to-input` | | Copy output labels to inputs (generator → echo). |
| `--copy-input-to-output` | | Copy input labels to outputs (recognizer → echo). |
| `--flank-input-wild` | | Add flanking delete states for partial input matching. |
| `--flank-output-wild` | | Add flanking insert states for partial output matching. |
| `--flank-either-wild` | | Add flanking insert or delete states: partially match either input or output at each end. |
| `--flank-both-wild` | | Add flanking insert & delete states: partially match input and/or output. |
| `--flank-input-geom EXPR` | | Like `--flank-input-wild`, but with geometrically-distributed flanking length. |
| `--flank-output-geom EXPR` | | Like `--flank-output-wild`, but with geometrically-distributed flanking length. |
| `--weight-input EXPR` | | Multiply input weights by expression (`%` = input symbol, `#` = alphabet size). |
| `--weight-output EXPR` | | Multiply output weights by expression. |
| `--weight-input-geom EXPR` | | Place geometric distribution over input length. |
| `--weight-output-geom EXPR` | | Place geometric distribution over output length. |
| `--strip-names` | | Remove state names (can speed up composition). |
| `--pad` | | Add dummy start & end states. |

## Infix Operators (Combining Two Machines)

| Option | Opcode | Description | Analogy |
|---|---|---|---|
| `--compose` | `=>` | Compose two machines (output of first feeds input of second). | Matrix multiplication |
| `--concatenate` | `.` | Concatenate two machines. | String concatenation |
| `--intersect` | `&&` | Intersect (shared input, pointwise product). | Pointwise product |
| `--union` | `\|\|` | Union (paths through one or the other). | Pointwise sum |
| `--loop` | `?+` | Loop: `x(yx)*`. | Kleene closure with spacer |
| `--flank` | | Flank: `y . x . y`. | |

If multiple machines are specified without an explicit combinator, **composition is implicit**.

## Grouping and Implicit Composition

Use `--begin` (`(`) and `--end` (`)`) to group sub-expressions.
Parentheses must be quoted on the command line to prevent shell interpretation.

```bash
boss --generate-uniform-dna \
  --concat --begin --preset protpsw --preset translate --end \
  --concat --generate-uniform-dna
```

This composes `protpsw` with `translate` inside the parentheses, then concatenates the result between two uniform DNA generators.

## Specifying Data

| Option | Description |
|---|---|
| `--input-chars SEQ` | Input sequence as command-line characters. |
| `--output-chars SEQ` | Output sequence as command-line characters. |
| `--input-fasta FILE` | Input from a FASTA file. |
| `--output-fasta FILE` | Output from a FASTA file. |
| `--input-json FILE` | Input from a JSON sequence file. |
| `--output-json FILE` | Output from a JSON sequence file. |
| `--data FILE` | Pairs of input & output sequences from a JSON file. |
| `--params FILE` | Load parameter values from a JSON file. |
| `--functions FILE` | Load function/constant definitions from a JSON file. |
| `--constraints FILE` | Load normalization constraints from a JSON file. |

## Inference Algorithms

| Option | Algorithm | Description |
|---|---|---|
| `--loglike` | Forward | Calculate log-likelihood by summing over all paths. |
| `--viterbi` | Viterbi | Calculate the log-weight of the most likely path. |
| `--align` | Viterbi | Find the most likely alignment (traceback). |
| `--counts` | Forward-Backward | Posterior expected parameter usage counts. |
| `--train` | Baum-Welch | Fit parameters using EM (via GSL optimizers). |
| `--beam-decode` | Beam search | Find the most likely *input* given an output. |
| `--beam-encode` | Beam search | Find the most likely *output* given an input. |
| `--prefix-decode` | CTC prefix search | Find the most likely input by prefix search. |
| `--prefix-encode` | CTC prefix search | Find the most likely output by prefix search. |
| `--viterbi-decode` | Viterbi | Decode input via Viterbi traceback. |
| `--viterbi-encode` | Viterbi | Encode output via Viterbi traceback. |
| `--cool-decode` | Simulated annealing | Decode input by simulated annealing. |
| `--mcmc-decode` | MCMC | Decode input by MCMC search. |
| `--random-encode` | Stochastic | Sample a random output by stochastic prefix search. |

| Option | Description |
|---|---|
| `--beam-width N` | Beam width for beam search (default 100). |
| `--prefix-backtrack N` | Max backtracking length for CTC prefix search. |
| `--decode-steps N` | Annealing steps per initial symbol. |
| `--seed N` | Random number seed. |
| `--wiggle-room N` | Allowed departure from training alignment. |
| `--use-defaults` | Use default values for unbound parameters. |

## Code Generation

Machine Boss can generate standalone C++, JavaScript, or WGSL (WebGPU) code implementing the Forward algorithm for a given machine.

| Option | Description |
|---|---|
| `--codegen DIR` | Generate parser code, save to the specified directory. |
| `--cpp64` | Generate C++ code (64-bit). |
| `--cpp32` | Generate C++ code (32-bit). |
| `--js` | Generate JavaScript code. |
| `--wgsl` | Generate WGSL compute shader and ES module for WebGPU. |
| `--inseq TYPE` | Input sequence type: `String`, `Intvec`, or `Profile`. |
| `--outseq TYPE` | Output sequence type: `String`, `Intvec`, or `Profile`. |
| `--showcells` | Include debugging output in generated code. |
| `--compileviterbi` | Compile Viterbi instead of Forward. |

## Output Options

| Option | Description |
|---|---|
| `--save FILE` | Save machine to file (instead of stdout). |
| `--graphviz` | Output in GraphViz DOT format. |
| `--stats` | Show model statistics (states, transitions, parameters). |
| `--evaluate` | Evaluate all transition weights numerically. |
| `--define-exprs` | Define and re-use repeated sub-expressions for compactness. |
| `--show-params` | Show unbound parameters in the final machine. |
| `--name-states` | Use state id (not number) to identify transition destinations. |

## General Options

| Option | Description |
|---|---|
| `-h, --help` | Display help message. |
| `-v, --verbose N` | Verbosity level (default 2). Higher values produce more log output. |
| `-d, --debug FUNC` | Log output from the specified function. |
| `-b, --monochrome` | Disable colored log output. |

## Examples

### Build a regex recognizer for an N-glycosylation motif

```bash
boss --aa-regex '^N[^P][ST][^P]$' --transpose >PS00001.json
```

### Calculate log-likelihood of a sequence containing a motif

```bash
boss --generate-uniform-dna . --generate-chars ACGCGT . --generate-uniform-dna \
  --output-chars AAGCAACGCGTAATA --loglike
```

### Encode binary data as non-repeating DNA

```bash
boss --preset bintern --preset terndna --input-chars 1010101 --beam-encode
```

### Decode DNA back to binary

```bash
boss --preset bintern --preset terndna --output-chars CGATATGC --beam-decode
```

### Base-call a neural network output

```bash
boss --recognize-csv output.csv --beam-decode
```

### Train a binary symmetric channel

```bash
boss bitnoise.json -N constraints.json -D seqpairs.json -T
```

### Compose two machines and visualize

```bash
boss --preset protpsw --preset translate --graphviz | dot -Tpng -o model.png
```

### Build a local DNA search model

```bash
boss --generate-uniform-dna \
  --concat --begin --preset protpsw --preset translate --end \
  --concat --generate-uniform-dna
```
