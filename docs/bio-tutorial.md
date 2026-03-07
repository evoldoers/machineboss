---
title: Biological Examples
nav_order: 11
permalink: /bio-tutorial/
---

# Tutorial: Biological Sequence Analysis

This tutorial demonstrates Machine Boss on biological sequence analysis tasks:
protein motif searching, profile HMM matching, evolutionary models, protein-to-DNA alignment,
and neural transducers.
For the introductory casino and reporter examples, see the [main tutorial](/tutorial/).

## Table of Contents
{: .no_toc }

1. TOC
{:toc}

## Protein Motif Search with Regular Expressions

Machine Boss can convert amino acid regular expressions
(such as [ProSite](https://prosite.expasy.org/) motifs) into weighted transducers,
then use the Viterbi algorithm to scan sequences for matches.

### Building a motif recognizer

The N-glycosylation motif (ProSite [PS00001](https://prosite.expasy.org/PS00001))
has the pattern `N-{P}-[ST]-{P}`, meaning:
asparagine, then any amino acid except proline, then serine or threonine, then any non-proline.

In regex syntax this is `N[^P][ST][^P]`. Let's build a recognizer:

```bash
boss --aa-regex 'N[^P][ST][^P]' >PS00001.json
```

This creates a transducer that reads an amino acid sequence as input
and emits the same sequence as output, with weight 1 wherever the motif matches
and weight 0 elsewhere.

### Scanning a protein sequence

The HIV-1 GP120 envelope glycoprotein contains several N-glycosylation sites.
We can scan for them using the Viterbi algorithm:

```bash
boss --input-fasta gp120.fa PS00001.json --viterbi
```

This outputs a JSON alignment showing where the motif matches.
The Viterbi log-likelihood will be zero (log 1), confirming that the motif is present.

For a protein without N-glycosylation sites (e.g. the Trp-cage miniprotein),
the log-likelihood will be negative infinity:

```bash
boss --input-fasta trp-cage.fa PS00001.json --viterbi
```

### DNA-level motif search

To search for a protein motif at the DNA level, compose the motif recognizer
with the genetic code. The `--transpose` flag flips the transducer so that
the DNA sequence is treated as the observed output:

```bash
boss PS00001.json --transpose --preset translate >PS00001-dna.json
```

This composite transducer reads DNA and matches wherever a translated reading frame
contains the N-glycosylation motif.

## Profile HMMs with HMMER

Machine Boss imports [HMMER3](http://hmmer.org/) profile HMMs
and converts them to its transducer format.
This allows profile HMMs to be composed with other machines.

### Importing a profile

Starting from a Pfam model (e.g. PF00516, the fibronectin type III domain):

```bash
boss --hmmer PF00516.hmm >gp120-core.json
```

This creates a generator (output-only machine) representing the profile HMM core model.

### Building a log-odds scoring model

To turn the profile into a scoring model, divide emission weights by SwissProt background frequencies
and add flanking states for local alignment:

```bash
boss --hmmer PF00516.hmm \
     --weight-output '1/$$pSwissProt%' \
     --params SwissProtComposition.json \
     --flank-output-wild \
     >gp120.json
```

The `--weight-output '1/$$pSwissProt%'` divides each emission probability
by the corresponding SwissProt background frequency, converting to a log-odds ratio.
The `--flank-output-wild` flag adds geometric-length flanking states
that model the unaligned regions at each end.

### Scoring sequences

Against a true GP120 sequence, the Viterbi log-likelihood should be positive (a match):

```bash
boss gp120.json --output-fasta gp120.fa --viterbi
```

Against an unrelated sequence, the log-likelihood should be negative (no match):

```bash
boss gp120.json --output-fasta trp-cage.fa --viterbi
```

### Multi-hit matching

To allow zero, one, or more matches to the motif in a single sequence,
use `--loop` with flanking:

```bash
boss --hmmer PF00516.hmm \
     --weight-output '1/$$pSwissProt%' \
     --params SwissProtComposition.json \
     --loop --begin --generate-one-aa --kleene-plus --end \
     --flank-output-wild \
     >gp120-multihit.json
```

The `--loop` flag places the profile inside a Kleene-plus loop
with a separator requiring at least one amino acid between matches.

### Plan7 architecture

Machine Boss also supports the full HMMER Plan7 architecture,
which adds N-terminal flanking (N), C-terminal flanking (C),
and optional multi-hit looping (J) states:

```bash
boss --hmmer-plan7 PF00516.hmm        # single-hit Plan7
boss --hmmer-multihit PF00516.hmm     # multi-hit Plan7
```

The Plan7 flanking states emit background amino acid frequencies (SwissProt composition)
and use a geometric length distribution with a configurable expected length (default 400).

## Evolutionary Models

### Jukes-Cantor substitution

The Jukes-Cantor model is the simplest model of DNA sequence evolution.
It assumes all substitutions are equally likely.
The preset `jukescantor` provides it as a transducer:

```bash
boss --preset jukescantor
```

### TKF91 indel model

The TKF91 model (Thorne, Kishino, and Felsenstein, 1991) extends the substitution model
with insertions and deletions.
Machine Boss provides it as a two-part model:

```bash
# Generate an ancestral sequence
boss --preset tkf91root --generate-one 20
# Evolve along a branch
boss --preset tkf91root --preset tkf91branch --generate-one 20
```

The `tkf91root` generator produces an ancestral sequence;
`tkf91branch` evolves an input sequence to an output with insertions, deletions, and substitutions.

To compute the log-likelihood of an alignment under TKF91:

```bash
boss --preset tkf91root --preset tkf91branch \
     --input-chars ACGTACGT --output-chars ACGACGTT -L
```

## Protein-to-DNA Alignment

The `psw2dna` preset composes protein Smith-Waterman alignment with the genetic code,
enabling direct protein-to-DNA alignment:

```bash
boss --preset psw2dna --input-fasta protein.fa --output-fasta dna.fa --viterbi
```

### With introns

The `pswint` preset extends protein-to-DNA alignment with intron support.
It allows GT-AG spliced introns to appear within the alignment:

```bash
boss --preset pswint --input-fasta protein.fa --output-fasta genomic.fa --viterbi
```

This is useful for aligning proteins against genomic DNA where the coding sequence
may be interrupted by introns.

## Transducer Composition

A key strength of Machine Boss is the ability to compose transducers into pipelines.
For example, to build a profile HMM scorer that works at the DNA level:

```bash
# Step 1: Build protein profile with scoring
boss --hmmer PF00516.hmm \
     --weight-output '1/$$pSwissProt%' \
     --params SwissProtComposition.json \
     >profile.json

# Step 2: Compose with reverse translation and flanking
boss profile.json --preset prot2dna --flank-output-wild >profile-dna.json

# Step 3: Score a DNA sequence
boss profile-dna.json --output-fasta dna.fa --viterbi
```

Each `--preset` or machine file on the command line is composed (via transducer composition)
with the preceding machine. This lets you build arbitrarily deep pipelines
from simple building blocks.

## Neural Transducers

The Machine Boss JAX package supports **neural transducers**:
neural networks that produce per-position parameters for a WFST.
The Forward algorithm computes the log-likelihood,
and gradients flow back through the dynamic programming into the neural network weights.

This is implemented via the `ParameterizedMachine` and `neural_log_forward_tok` API.
A `ParameterizedMachine` is compiled from a `Machine` whose transitions use symbolic weight expressions
with named parameters (e.g. `t`, `pIns`, `pDel`).
At runtime, the caller supplies each parameter as a tensor of shape `(Li+1, 1)`, `(1, Lo+1)`, or `(Li+1, Lo+1)`,
allowing different parameter values at each position in the DP grid.

### Example 1: Neural DNA Copy Transducer

This example trains a 1D CNN to predict per-position evolutionary parameters
for a 6-state TKF91-like DNA copy transducer.

**Machine.** Six states (begin, wait, match, insert, delete, end) with Jukes-Cantor substitution.
Three free parameters per position: evolutionary distance `t`, insertion probability `pIns`,
and deletion probability `pDel`. The machine's `defs` derive substitution probabilities
from `t` via the JC model: `pNoSub = exp(-t)`, `pSame = pNoSub + (1-pNoSub)/4`.

```python
from machineboss.neural.dna_copy import make_dna_copy_machine
from machineboss.jax.jax_weight import ParameterizedMachine

machine = make_dna_copy_machine()       # 6-state WFST
pm = ParameterizedMachine.from_machine(machine)
print(pm.free_params)                   # {'t', 'pIns', 'pDel'}
```

**CNN.** A small Flax/Linen 1D CNN maps one-hot DNA input `(L, 4)` to three per-position outputs.
Output activations enforce valid ranges: `softplus` for `t` (positive), `sigmoid * 0.3` for probabilities.

```python
from machineboss.neural.dna_copy import DNACopyCNN, onehot_dna, cnn_params_to_dp_params

model = DNACopyCNN(hidden=32, kernel=5)
x = onehot_dna("ACGTACGT")              # (8, 4)
cnn_params = model.init(jax.random.PRNGKey(0), x)
t, pIns, pDel = model.apply(cnn_params, x)
dp_params = cnn_params_to_dp_params(t, pIns, pDel)  # {'t': (9,1), 'pIns': (9,1), ...}
```

**Training.** The loss is the negative log-likelihood from the Forward algorithm.
Gradients flow through the DP back into the CNN:

```python
from machineboss.jax.dp_neural import neural_log_forward_tok

def loss_fn(cnn_params, x, in_tok, out_tok):
    t, pIns, pDel = model.apply(cnn_params, x)
    dp_params = cnn_params_to_dp_params(t, pIns, pDel)
    return -neural_log_forward_tok(pm, in_tok, out_tok, dp_params)

grads = jax.grad(loss_fn)(cnn_params, x, in_tok, out_tok)
```

**Simulator.** The included DNA simulator generates ancestor-descendant pairs
with homopolymer-dependent error rates for training data:

```python
from machineboss.neural.simulator import simulate_dna_pair
ancestor, descendant = simulate_dna_pair(jax.random.PRNGKey(0), length=100,
                                          base_sub_rate=0.08, hp_multiplier=3.0)
```

A complete training script is provided in `examples/train_neural_dna_copy.py`.

### Example 2: Neural TKF92 Protein Transducer

This example trains an MSA transformer to predict per-position parameters
for a 7-state TKF92 protein evolution transducer.

**Machine.** Seven states (begin, orphan, wait, match, insert, delete, end) with F81 amino acid
substitution and TKF92 fragment extension.
The TKF92 model extends TKF91 with a fragment extension probability `r`:
match, insert, and delete states have self-loops with probability `r`,
allowing runs of operations within a single fragment.
Free parameters: `t`, `insRate`, `delRate`, `r`, and 20 equilibrium frequencies `pi_0`..`pi_19`
(24 total per position).

```python
from machineboss.neural.tkf92 import make_tkf92_machine

machine = make_tkf92_machine()           # 7-state WFST, 20-AA alphabet
pm = ParameterizedMachine.from_machine(machine)
print(len(pm.free_params))              # 24
```

**MSA Transformer.** A Flax/Linen transformer with row attention (within sequences)
and column attention (across sequences) reads a one-hot MSA `(N, L, 21)` and produces
a mean-pooled representation `(L, d_model)`:

```python
from machineboss.neural.msa_transformer import MSATransformer
from machineboss.neural.tkf92 import TKF92Heads, heads_to_dp_params

transformer = MSATransformer(d_model=64, n_heads=4, n_layers=2)
heads = TKF92Heads()

# msa_onehot: (N, L, 21) from Stockholm parser
embeddings = transformer.apply(t_params, msa_onehot)   # (L, 64)
t, insRate, delRate, r, pi = heads.apply(h_params, embeddings)
dp_params = heads_to_dp_params(t, insRate, delRate, r, pi)
```

The parameter heads enforce constraints: `t > 0`, `delRate > insRate` (required for TKF91),
`0 < r < 1`, and `sum(pi) = 1`.

**Stockholm parser.** A minimal parser reads Pfam-style `.sto` files:

```python
from machineboss.neural.stockholm import parse_stockholm_file
msa = parse_stockholm_file("PF00516.sto")
seq_i, seq_j = msa.ungapped_pair(0, 1)  # extract a training pair
```

A complete training script is provided in `examples/train_neural_tkf92.py`.

### Notes on practical training

**Sequence embeddings.**
In practice, the MSA transformer shown here would benefit substantially from
pre-trained protein language model embeddings as input features.
[ESM-2](https://github.com/facebookresearch/esm) or
[ESM-MSA-1b](https://github.com/facebookresearch/esm) embeddings of the MSA rows
could replace (or augment) the raw one-hot input,
providing rich per-residue representations learned from millions of protein sequences.
This would likely improve convergence and generalization,
especially for small families where the MSA alone provides limited signal.

**Train/test splitting.**
A proper evaluation of this setup should use tree-based splits
of Pfam or TreeFam families to avoid data leakage.
Specifically, the predictive target sequence (and any close homologs)
should be excluded from the MSA used to supply embeddings to the transducer.
Phylogeny-aware splitting ensures that the model cannot simply memorize
the target from highly similar sequences in the conditioning MSA.
Without such precautions, performance estimates will be inflated,
as the model can exploit near-identical sequences rather than learning
genuine evolutionary patterns.
