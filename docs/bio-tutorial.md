---
title: Biological Examples
nav_order: 11
permalink: /bio-tutorial/
---

# Tutorial: Biological Sequence Analysis

This tutorial demonstrates Machine Boss on biological sequence analysis tasks:
protein motif searching, profile HMM matching, evolutionary models, and protein-to-DNA alignment.
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
