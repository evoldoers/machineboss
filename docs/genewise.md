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

## A Brief History of Transducers

The weighted finite-state transducer has a distinguished intellectual lineage
that spans computing, linguistics, cryptography, and biology.

The finite-state machine itself dates to the foundational work of
Moore (1956) and Mealy (1955), who formalized sequential circuits
as state machines with input and output.
These automata became central to formal language theory
(Rabin and Scott, 1959; Hopcroft and Ullman, 1979)
and to the theory of computation more broadly —
a lineage that traces back, of course, to Turing's original 1936 paper
and, before that, to Babbage's mechanical engines.
It is no accident that the earliest computers were built by cryptographers:
Turing at Bletchley Park, and Zuse, Colossus, and ENIAC
all emerged from wartime code-breaking and ballistics.

Weighted transducers found their modern mathematical footing
in the work of Mehryar Mohri and colleagues at AT&T Bell Labs in the 1990s
(Mohri, 1997; Mohri, Pereira, and Riley, 2002),
who developed efficient algorithms for composition, determinization,
and minimization of weighted finite-state transducers,
and applied them at scale to speech recognition and natural language processing.
Their OpenFst library remains a standard reference implementation.

In machine learning, the transducer concept re-emerged in neural sequence-to-sequence models.
Graves (2012) introduced the Recurrent Neural Network Transducer (RNN-T),
which combines a transcription network (analogous to the input model)
with a prediction network (analogous to the language model)
via a joint network that plays the role of the transducer.
RNN-T and its successors are now the dominant architecture
for streaming speech recognition.

In computational biology, hidden Markov models and pair HMMs
have been workhorses since the 1980s
(Churchill, 1989; Krogh et al., 1994; Durbin et al., 1998).
Profile HMMs, formalized by Krogh et al. (1994) and implemented in Sean Eddy's HMMER,
became the standard representation for protein families.
But the most ambitious biological transducer — the one that pushed
the state machine concept furthest — was GeneWise.

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

### Parallel architectures and hardware acceleration

The Smith-Waterman and profile HMM algorithms at the heart of GeneWise
were natural targets for parallelization.
In the late 1990s and 2000s, these algorithms were ported to various parallel architectures,
most notably the systolic array processors built by Paracel (later acquired by Celera).
Paracel held several patents on hardware-accelerated sequence comparison
and sold specialized machines to genomics centers.
They were also rumored to be heavily used by government intelligence agencies
for DNA database searching —
another episode in the long and intertwined history of bioinformatics,
computational linguistics, and signals intelligence.

This is not as surprising as it might seem.
Sequence alignment, speech recognition, and cryptanalysis
all reduce to the same mathematical problem:
finding the most probable path through a hidden Markov model
or the highest-scoring alignment between two symbol sequences.
The Viterbi algorithm (Viterbi, 1967), originally developed for decoding
convolutional codes in telecommunications,
is the same algorithm used to find optimal alignments in biology
and to decode speech in natural language processing.
From Babbage and Turing through Paracel's classified contracts,
the history of computing, cryptography, and sequence analysis
has been one long conversation.

### Dynamite: a precursor to Machine Boss

GeneWise was built using Dynamite (Birney, 1997),
a domain-specific language and code generator for dynamic programming.
Dynamite took a high-level declarative description of a DP recurrence —
specifying states, transitions, and how component models composed —
and compiled it into efficient C code.
This is a clear precursor to Machine Boss's own approach
of representing models as composable JSON transducers
and compiling them to optimized C++, JavaScript, or WebGPU code.

Dynamite itself could target multiple platforms and generated code
for some quite esoteric architectures of its era.
Machine Boss continues this tradition with its multi-backend code generator
(`boss --cpp64 --codegen` for C++, `boss --js --codegen` for JavaScript,
and WebGPU shader generation for GPU execution).

### From GeneWise to Exonerate to miniprot

GeneWise led to Exonerate (Slater and Birney, 2005),
a more general pairwise alignment tool
that extended the GeneWise approach with additional alignment models
and heuristic seeding for faster whole-genome searches.
Exonerate became widely used for protein-to-genome alignment
in annotation pipelines.

More recently, Heng Li's miniprot (Li, 2023) revisited the protein-to-genome
alignment problem with modern algorithmic techniques,
achieving dramatic speedups through chaining-based seeding
and a simplified splice-aware alignment model.
The miniprot paper cites and benchmarks against both GeneWise and Exonerate,
placing itself as the latest chapter in this lineage.

Machine Boss's fused DP kernels bring this story into the GPU era.
Harnessing the power of JAX's JIT compilation, automatic differentiation,
and GPU acceleration — and developed with the assistance of coding agents —
we offer the latest chapter in the ongoing story
of transducers, composition, and biological sequence analysis.

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

### Transducers and automata theory

- Moore, E. F. (1956).
  Gedanken-experiments on sequential machines.
  *Automata Studies*, Princeton University Press, 129–153.

- Mealy, G. H. (1955).
  A method for synthesizing sequential circuits.
  *Bell System Technical Journal*, 34(5), 1045–1079.

- Mohri, M. (1997).
  Finite-state transducers in language and speech processing.
  *Computational Linguistics*, 23(2), 269–311.

- Mohri, M., Pereira, F., & Riley, M. (2002).
  Weighted finite-state transducers in speech recognition.
  *Computer Speech & Language*, 16(1), 69–88.

- Graves, A. (2012).
  Sequence transduction with recurrent neural networks.
  *arXiv:1211.3711*.

### Biological sequence analysis

- Durbin, R., Eddy, S. R., Krogh, A., & Mitchison, G. (1998).
  *Biological Sequence Analysis: Probabilistic Models of Proteins and Nucleic Acids*.
  Cambridge University Press.

- Krogh, A., Brown, M., Mian, I. S., Sjolander, K., & Haussler, D. (1994).
  Hidden Markov models in computational biology: applications to protein modeling.
  *Journal of Molecular Biology*, 235(5), 1501–1531.

- Eddy, S. R. (1998).
  Profile hidden Markov models.
  *Bioinformatics*, 14(9), 755–763.

### GeneWise, Dynamite, and successors

- Birney, E., & Durbin, R. (1997).
  Dynamite: a flexible code generating language for dynamic programming methods used in sequence comparison.
  *Proceedings of the Fifth International Conference on Intelligent Systems for Molecular Biology*, 56–64.

- Birney, E., & Durbin, R. (2000).
  Using GeneWise in the _Drosophila_ annotation experiment.
  *Genome Research*, 10(4), 547–548.

- Birney, E., Clamp, M., & Durbin, R. (2004).
  GeneWise and Genomewise.
  *Genome Research*, 14(5), 988–995.
  [doi:10.1101/gr.1865504](https://doi.org/10.1101/gr.1865504)

- Slater, G. St. C., & Birney, E. (2005).
  Automated generation of heuristics for biological sequence comparison.
  *BMC Bioinformatics*, 6, 31.
  [doi:10.1186/1471-2105-6-31](https://doi.org/10.1186/1471-2105-6-31)

- Li, H. (2023).
  Protein-to-genome alignment with miniprot.
  *Bioinformatics*, 39(1), btad014.
  [doi:10.1093/bioinformatics/btad014](https://doi.org/10.1093/bioinformatics/btad014)
