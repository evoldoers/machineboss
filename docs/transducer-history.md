---
title: A Brief History of Transducers
nav_order: 14
permalink: /transducer-history/
---

# A Brief History of Transducers

Notes on the lineage of state machines in bioinformatics and beyond.

## Table of Contents
{: .no_toc }

1. TOC
{:toc}

## Automata and formal language theory

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

## Weighted transducers in speech and NLP

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

## HMMs and profile HMMs in biology

In computational biology, hidden Markov models and pair HMMs
have been workhorses since the 1980s
(Churchill, 1989; Krogh et al., 1994; Durbin et al., 1998).
The promotion of HMMs in biological sequence analysis
owes much to the group at Cambridge and the nearby Sanger Centre (Hinxton)
in the 1990s:
Richard Durbin, Anders Krogh, and Graeme Mitchison
(with Sean Eddy) wrote the influential textbook
_Biological Sequence Analysis_ (Durbin et al., 1998),
which became the standard reference for probabilistic methods in bioinformatics.
Krogh et al. (1994) formalized profile HMMs for protein families,
and Sean Eddy's HMMER software made them a practical tool
for large-scale protein sequence search —
work that continues to underpin databases like Pfam and InterPro.
Without HMMER's profile HMMs, GeneWise would not have existed.

But the most ambitious biological transducer — the one that pushed
the state machine concept furthest — was GeneWise.

## GeneWise: the most elaborate state machine in bioinformatics

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
Birney used _Dynamite_, a code generator he had begun developing
before arriving at the Sanger Centre, that took a declarative description
of the composed DP recurrence and generated optimized C code
for the fused algorithm.
The generated code interleaved the recurrences of the component models,
visiting only the reachable states of the product.

It is said that Birney, who did the majority of his work on GeneWise and GenomeWise
at the Sanger Centre and was already publishing on the topic as an undergraduate,
was notorious among his peers for having cited his own 1996
"PairWise and SearchWise" paper (Birney, 1996) in his undergraduate final examination.

The resulting GeneWise program became a cornerstone of eukaryotic genome annotation
and was used extensively in the annotation of the human genome.

## Parallel architectures and hardware acceleration

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

## Dynamite: a precursor to Machine Boss

GeneWise was built using Dynamite (Birney, 1997),
a domain-specific language and code generator for dynamic programming.
Dynamite took a high-level declarative description of a DP recurrence —
specifying states, transitions, and how component models composed —
and compiled it into efficient C code.
This is a clear precursor to Machine Boss's own approach
of representing models as composable JSON transducers
and compiling them to optimized C++, JavaScript, or WebGPU code.

Crucially, Dynamite was a _generic_ DP code generator:
it understood recurrence structure (states, transitions, scores)
but had no built-in notion of probabilistic models.
It could generate Viterbi and Forward-like score computations,
but not training algorithms such as Forward-Backward or Baum-Welch,
because those require understanding the probabilistic semantics —
that transition weights are probabilities, that the goal is to maximize likelihood,
and that expected counts can be computed by combining forward and backward passes.
As a result, GeneWise could _score_ and _align_ but could not _train_ its own parameters.

Dynamite itself could target multiple platforms and generated code
for some quite esoteric architectures of its era.
Machine Boss continues this tradition with its multi-backend code generator
(`boss --cpp64 --codegen` for C++, `boss --js --codegen` for JavaScript,
and WebGPU shader generation for GPU execution).

## HMMoC: a compiler for HMM algorithms

Gerton Lunter's HMMoC (Lunter, 2007) took the code-generation idea further
by building probabilistic semantics into the compiler.
HMMoC translated declarative HMM descriptions
into efficient C++ implementations not just of Viterbi and Forward,
but also Backward, Forward-Backward, and Baum-Welch training.
Because HMMoC understood that its models were probabilistic,
it could automatically generate the derivative computations
needed for parameter estimation — something Dynamite could not do.

Like Dynamite, HMMoC allowed researchers to specify complex HMM topologies
at a high level and have the compiler handle loop ordering, memory layout,
and numerical details.
But by closing the loop from scoring to training,
HMMoC demonstrated that a code generator for biological sequence models
could be a complete toolkit, not just an alignment engine.

## Generic HMM libraries

In parallel with these code-generation approaches,
many bioinformatics libraries offered generic HMM implementations
with built-in training algorithms —
far too many to enumerate here.
These ranged from general-purpose toolkits
(GHMM, HMMlib, pomegranate, hmmlearn, StochHMM, among others)
to domain-specific frameworks embedded in larger bioinformatics packages.
Most provided Forward-Backward and Baum-Welch out of the box,
but operated on explicitly instantiated state spaces
rather than generating compiled code for specific model topologies.
The trade-off was generality versus performance:
a generic library can handle any HMM,
but a compiled code generator like Dynamite, HMMoC, or Machine Boss
can exploit the specific structure of a model
for dramatically better cache behavior, loop ordering, and vectorization.

## From GeneWise to Exonerate to miniprot

GeneWise led to Exonerate (Slater and Birney, 2005),
a more general pairwise alignment tool
that extended the GeneWise approach with additional alignment models
and heuristic seeding for faster whole-genome searches.
Exonerate became widely used for protein-to-genome alignment
in annotation pipelines, though it traded some of GeneWise's
full probabilistic model for speed.

More recently, Heng Li's miniprot (Li, 2023) revisited the protein-to-genome
alignment problem with modern algorithmic techniques,
achieving dramatic speedups through chaining-based seeding
and a simplified splice-aware alignment model.
The miniprot paper cites and benchmarks against both GeneWise and Exonerate,
placing itself as the latest chapter in this lineage.

Neither Exonerate nor miniprot fully replicates the complete probabilistic model
that GeneWise offered: a Viterbi DP over the joint space of
profile HMM, genetic code, intron model, and frameshift model.
(GeneWise itself only implemented Viterbi, not Forward.)
Machine Boss's [fused DP kernels](/genewise/) go further,
supporting Forward, Viterbi, and Forward-Backward over the fused state space,
with fully differentiable likelihoods via JAX —
enabling neural HMMs whose parameters depend on the sequence.

Machine Boss's fused DP kernels bring this story into the GPU era.
Harnessing the power of JAX's JIT compilation, automatic differentiation,
and GPU acceleration — and developed with the assistance of coding agents —
we offer the latest chapter in the ongoing story
of transducers, composition, and biological sequence analysis.

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

### GeneWise, Dynamite, HMMoC, and successors

- Birney, E. (1996).
  PairWise and SearchWise: finding the optimal alignment in a simultaneous comparison of a protein profile against all DNA translation frames.
  *Unpublished undergraduate thesis*.

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

- Lunter, G. (2007).
  HMMoC — a compiler for hidden Markov models.
  *Bioinformatics*, 23(18), 2485–2487.
  [doi:10.1093/bioinformatics/btm350](https://doi.org/10.1093/bioinformatics/btm350)

- Slater, G. St. C., & Birney, E. (2005).
  Automated generation of heuristics for biological sequence comparison.
  *BMC Bioinformatics*, 6, 31.
  [doi:10.1186/1471-2105-6-31](https://doi.org/10.1186/1471-2105-6-31)

- Li, H. (2023).
  Protein-to-genome alignment with miniprot.
  *Bioinformatics*, 39(1), btad014.
  [doi:10.1093/bioinformatics/btad014](https://doi.org/10.1093/bioinformatics/btad014)

### Machine Boss

- Silvestre-Ryan, J., Wang, Y., Sharma, M., Lin, S., Shen, Y., Dider, S., & Holmes, I. (2021).
  Machine Boss: rapid prototyping of bioinformatic automata.
  *Bioinformatics*, 37(1), 142–143.
  [doi:10.1093/bioinformatics/btaa633](https://doi.org/10.1093/bioinformatics/btaa633)
