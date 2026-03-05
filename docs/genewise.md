---
title: GeneWise-style Models
nav_order: 13
permalink: /genewise/
---

# Tutorial: GeneWise-style Fused DP

This tutorial explains how Machine Boss supports GeneWise-style protein-to-DNA alignment
using fused dynamic programming kernels in the Python/JAX backend.
Instead of explicitly composing a profile HMM with a coding transducer
(which can produce a very large state space),
the fused approach interleaves the two models' DP in a single pass.

## Table of Contents
{: .no_toc }

1. TOC
{:toc}

## Background: Composition vs Fusion

### The composition approach

In Machine Boss, transducer composition is the standard way to build multi-layer models.
For example, to score a DNA sequence against a protein profile HMM:

```bash
boss --hmmer-plan7 profile.hmm --compose transducer.json --output-chars ACGT... -L
```

This explicitly constructs the composed machine (profile HMM × transducer),
then runs the Forward algorithm on the result.
The composed machine's state space is the Cartesian product of the two models' state spaces:
if the profile has _S₁_ states and the transducer has _S₂_ states,
the composed machine has up to _S₁ × S₂_ states.

For large profile HMMs (hundreds of match states) composed with the genetic code
(64 codons × reading frame states), this product can be very large,
making the composed machine expensive to construct and slow to evaluate.

### GeneWise: the most elaborate state machine in bioinformatics

GeneWise, developed by Ewan Birney and Richard Durbin at the Sanger Centre
in the late 1990s, was arguably the most complex dynamic programming state machine
that computational biology had seen up to that point.
It was the first practical tool to align a protein sequence
(or protein profile HMM) directly against genomic DNA,
accounting simultaneously for:

- **The protein model**: a full profile HMM with match, insert, and delete states
  at every position in the protein family.
- **The genetic code**: all 64 codons and the mapping from codon triplets to amino acids,
  including synonymous codons.
- **Introns**: GT-AG splice donor and acceptor signals,
  variable-length intronic sequence between coding exons,
  and the requirement that introns preserve the reading frame.
- **Sequencing errors**: frameshifts from single-base insertions or deletions
  in the genomic DNA, which shift the reading frame but can be recovered.

No previous tool had tried to handle all of these simultaneously in a single DP.
Earlier approaches either aligned protein to cDNA (no introns)
or used heuristic multi-stage pipelines that first predicted exons
and then aligned proteins.
GeneWise unified everything into one mathematically coherent model.

The trick was to recognize that all these components are themselves state machines
(weighted finite-state transducers) that can be composed —
but rather than explicitly building the enormous product machine
and then running DP on it,
Birney wrote a code generator called _Dynamite_ that took a declarative description
of the composed DP recurrence and generated optimized C code
for the fused algorithm.
The generated code interleaved the recurrences of the component models,
visiting only the reachable states of the product.

It is said that Birney, who began this work as a PhD student at the Sanger Centre
and was already publishing on the topic as an undergraduate,
was notorious among his peers for having cited his own 1996
"PairWise and SearchWise" paper in his undergraduate final examination.

The resulting GeneWise program became a cornerstone of eukaryotic genome annotation
and was used extensively in the annotation of the human genome.

### The key insight

GeneWise avoids explicit composition
by interleaving the two models' DP recurrences.
At each step, the algorithm simultaneously advances through
the profile HMM states and the coding transducer states,
computing the joint probability without materializing the full product machine.

The observation is that the profile HMM and transducer interact only
through their shared intermediate alphabet (amino acids):
the profile emits an amino acid, and the transducer consumes it and produces DNA.
This coupling can be computed on the fly during DP,
without ever building the composed state space.

Machine Boss implements this idea in its Python/JAX backend
as the **fused DP kernel**.

## The Fused Kernel

The fused kernel is implemented in `python/machineboss/jax/fused.py`.
It takes two machines:

1. A **generator** (Plan7 profile HMM): output-only, emits intermediate symbols (amino acids).
2. A **transducer**: reads intermediate symbols as input, produces final output (DNA).

The composite state is the pair (profile_state, transducer_state).
The DP scans over the output sequence (DNA) using `jax.lax.scan`,
and at each position considers all ways the profile can emit an intermediate symbol
that the transducer then consumes while producing the observed output.

```python
from machineboss.hmmer import HmmerModel
from machineboss.eval import EvaluatedMachine
from machineboss.machine import Machine
from machineboss.jax.fused import FusedMachine, fused_log_forward

# Build the two machines
hmmer = HmmerModel.read(open("profile.hmm"))
plan7_machine = hmmer.plan7_machine()
plan7_em = EvaluatedMachine.from_machine(plan7_machine)

transducer = Machine.from_file("transducer.json")
td_em = EvaluatedMachine.from_machine(transducer)

# Build fused representation
fused = FusedMachine.build(plan7_em, td_em)

# Run Forward algorithm
import jax.numpy as jnp
output_seq = jnp.array(td_em.tokenize_output(list("ACGTACGT")))
log_likelihood = fused_log_forward(fused, output_seq)
```

### Semiring parameterization

The fused kernel is parameterized by a semiring, allowing the same code
to compute either Forward (log-sum-exp semiring) or Viterbi (max-plus semiring) scores:

```python
from machineboss.jax.fused import fused_log_viterbi

viterbi_score = fused_log_viterbi(fused, output_seq)
```

### Exact equivalence with composition

The fused kernel computes exactly the same result as explicit composition
followed by the standard Forward or Viterbi algorithm.
This is verified by tests that compare the fused output against
`boss --hmmer-plan7 ... --compose ... -L` for the same inputs
(see `python/machineboss/jax/test/test_fused.py` and `test_fused_plan7.py`).

## Plan7-aware Fused Kernel

The generic fused kernel (`FusedMachine`) treats the profile HMM as an opaque generator
with _S₁_ states, giving O(_S₁ × S₂_) work per output position.

The Plan7-aware kernel (`FusedPlan7Machine` in `python/machineboss/jax/fused_plan7.py`)
exploits the linear chain structure of the HMMER Plan7 architecture
to reduce this to O(_K × S₂_) per position,
where _K_ is the number of profile nodes (match positions).

### Architecture

The Plan7-aware kernel uses nested scans:

- **Outer scan** (`jax.lax.scan`): iterates over output sequence positions.
- **Inner scan** (`jax.lax.scan`): iterates over core profile nodes _k = 1..K_.

At each output position, the algorithm:
1. **Emits**: Each core state (M_k, I_k) and flanking state (N, C, J)
   emits an amino acid, which the transducer consumes while producing the observed output base.
2. **Routes**: Post-emission routing propagates probability mass through the Plan7 topology
   (M→M, M→I, M→D, I→M, I→I, D→M, D→D chains) using the inner scan.
3. **Flanking**: Handles N/C/J flanking states and E→B looping (for multi-hit mode) separately.

The inner scan carries only O(_S₂_) state (the transducer state vector)
through the profile nodes, rather than the full O(_S₁ × S₂_) composite state.

```python
from machineboss.jax.fused_plan7 import (
    FusedPlan7Machine,
    fused_plan7_log_forward,
    fused_plan7_log_viterbi,
)

# Build directly from HmmerModel (not via generic Machine)
fm = FusedPlan7Machine.build(hmmer, td_em, multihit=False)
ll = fused_plan7_log_forward(fm, output_seq)
```

### Multi-hit mode

The Plan7-aware kernel supports both single-hit and multi-hit modes:

```python
# Single-hit: one match per sequence
fm = FusedPlan7Machine.build(hmmer, td_em, multihit=False)

# Multi-hit: allows multiple matches with J-state looping
fm = FusedPlan7Machine.build(hmmer, td_em, multihit=True)
```

In multi-hit mode, the E state routes to both C (terminal) and J (loop back to B)
with equal probability, and the J state emits background amino acids
like the N and C flanking states.

### Equivalence testing

The Plan7-aware kernel is tested for exact equivalence with the generic fused kernel
(which in turn is tested against explicit composition via the `boss` CLI):

```
test_fused_plan7.py::TestFusedPlan7Forward::test_vs_boss_compose
    Plan7 fused == boss --hmmer-plan7 ... --compose ... -L

test_fused_plan7.py::TestFusedPlan7Forward::test_fn3_bitecho
    Plan7 fused == generic fused (FusedMachine)
```

Both kernels produce identical log-likelihoods (within numerical tolerance)
for the same inputs, confirming that the Plan7-specific optimizations
do not change the mathematical result.

## When to Use Each Approach

| Approach | Use when |
|----------|----------|
| Explicit composition (`boss --compose`) | Small machines, or when you need the composed machine as a JSON file |
| Generic fused (`FusedMachine`) | Moderate-sized generators with any transducer |
| Plan7-aware fused (`FusedPlan7Machine`) | Large HMMER profiles — exploits linear chain structure for efficiency |

The fused kernels are JIT-compiled by JAX, making them suitable for
GPU acceleration and gradient computation (for differentiable sequence analysis).

## References

- Birney, E., & Durbin, R. (1997).
  Dynamite: a flexible code generating language for dynamic programming methods used in sequence comparison.
  *Proceedings of the Fifth International Conference on Intelligent Systems for Molecular Biology*, 56–64.

- Birney, E., Clamp, M., & Durbin, R. (2004).
  GeneWise and Genomewise.
  *Genome Research*, 14(5), 988–995.
  [doi:10.1101/gr.1865504](https://doi.org/10.1101/gr.1865504)
