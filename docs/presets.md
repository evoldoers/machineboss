---
title: Preset Machines
nav_order: 10
permalink: /presets/
---

# Preset Machines

Machine Boss ships with a library of preset machines that can be loaded with the `--preset` flag.
Multiple presets can be chained on one command line to build composite pipelines via transducer composition.

## Table of Contents
{: .no_toc }

1. TOC
{:toc}

## Using Presets

```bash
boss --preset NAME              # load preset by name
boss --preset A --preset B      # compose two presets: A then B
```

To inspect a preset's JSON definition:
```bash
boss --preset NAME              # prints JSON to stdout
```

## Sequence Alignment

| Preset | Description |
|--------|-------------|
| `dnapsw` | DNA pairwise alignment (Smith-Waterman) with substitution and gap parameters |
| `dnapsw_mix2` | DNA alignment with a two-component mixture model |
| `dnapswnbr` | DNA alignment with neighbor-dependent substitution context |
| `protpsw` | Protein pairwise alignment (Smith-Waterman) with 20-amino-acid substitution matrix |
| `psw2dna` | Protein-to-DNA alignment: protein Smith-Waterman composed with reverse translation |
| `pswint` | Protein Smith-Waterman with intron support (allows GT-AG spliced introns) |

These are transducers that read one sequence as input and write an aligned sequence as output.
They include parameterized gap-open and gap-extend penalties, substitution weights,
and (for `psw2dna` and `pswint`) the genetic code.

## Evolutionary Models

| Preset | Description |
|--------|-------------|
| `jukescantor` | Jukes-Cantor model of DNA sequence divergence |
| `tkf91root` | TKF91 root model: generates an ancestral sequence with indel and substitution rates |
| `tkf91branch` | TKF91 branch model: evolves a sequence along a phylogenetic branch |

The TKF91 (Thorne, Kishino, Felsenstein 1991) model is a continuous-time Markov model of
sequence evolution with insertions, deletions, and substitutions.
`tkf91root` is a generator (output only) that produces the root sequence;
`tkf91branch` is a transducer that evolves an input sequence to an output sequence.
Together they model pairwise sequence evolution.

## Genetic Code and Translation

| Preset | Description |
|--------|-------------|
| `translate` | Standard genetic code: DNA codons to amino acids (64 codon mappings) |
| `prot2dna` | Reverse translation: protein to DNA via composition with `translate` inverse |
| `simple_introns` | Intron insertion transducer with GT-AG splice signals |
| `add-intron-placeholders` | Marks intron positions in aligned sequences with placeholder labels |
| `pint` | Protein identity transducer with optional 3-symbol intron emission |

## DNA Storage Coding

| Preset | Description |
|--------|-------------|
| `bintern` | Binary to ternary: 5 bits → 2 ternary digits (rate 5/4 bits per trit) |
| `terndna` | Ternary to non-repeating DNA: each trit selects from 3 bases that differ from the previous one |
| `nontern` | Composition of `bintern` and `terndna`: bit blocks directly to non-repeating ternary-DNA tokens |
| `bitcod` | Trivial binary-to-ternary padding: bit `b` → `b12` (rate 1/3) |
| `bitbase` | Binary to DNA directly: 1 bit → 1 base, selecting from 3 alternatives (rate 1 bit/base) |
| `bytern` | Byte-oriented binary to ternary: 8 bits → 3 pairs of ternary digits |
| `AfeI` | Like `terndna` but avoids the AfeI restriction enzyme recognition site AGCGCT |

The `bintern` → `terndna` pipeline implements the DNA storage code described by
Goldman et al. (2013). See the [DNA Storage tutorial](/dna-storage/) for details.

## Error-Correcting Codes

| Preset | Description |
|--------|-------------|
| `hamming31` | (3,1) repetition code: each input bit is repeated 3 times |
| `hamming74` | (7,4) Hamming code: 4 input bits → 7-bit codeword with 3 parity bits |

These generate error-correcting codewords as transducers.
They can be composed with channel noise models (e.g. a bit-flip transducer)
and decoded using beam search.

## Alphabet Conversion

| Preset | Description |
|--------|-------------|
| `compdna` | DNA complement (A↔T, C↔G) with IUPAC ambiguity code support |
| `comprna` | RNA complement (A↔U, C↔G) with IUPAC ambiguity code support |
| `dna2rna` | DNA to RNA: T → U |
| `rna2dna` | RNA to DNA: U → T |
| `tolower` | Converts all ASCII characters to lowercase |
| `toupper` | Converts all ASCII characters to uppercase |

## Alphabet Expansion

| Preset | Description |
|--------|-------------|
| `iupacdna` | Expands IUPAC DNA ambiguity codes (e.g. R → A or G, N → any base) |
| `iupacaa` | Expands IUPAC protein codes (X → any amino acid, B → D or N, etc.) |

These are useful when input sequences contain ambiguity codes that need to be
resolved before further processing.

## Helper Machines

| Preset | Description |
|--------|-------------|
| `null` | Empty machine with a single state and no transitions |
| `flankbase` | Generator: emits geometric-length runs of `base` symbols |
| `geom_iid_binary` | Generator: emits geometric-length runs of `0`/`1` with parameter `p` |
| `base2acgt` | Maps abstract `base` tokens to concrete ACGT nucleotides with fitted probabilities |

These are building blocks used internally or in composed pipelines.
`flankbase` is used with `--flank-output-wild` to add flanking regions around profile HMM matches.
`base2acgt` resolves the abstract `base` symbol used by some machines into concrete nucleotides.
