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
| `tkf91-root-dna-jc` | TKF91 root: geometric DNA singlet, P(L=k)=κ^k(1−κ) with κ=insRate/delRate |
| `tkf91-branch-dna-jc` | TKF91 branch transducer: 7-state DNA + Jukes-Cantor substitutions |
| `tkf92-root-prot-f81` | TKF92 root: ν-modified geometric protein singlet, P(L=0)=1−κ, P(L≥1)=κ·ν^(k−1)·(1−ν), with ν=`r`. F81 equilibrium |
| `tkf92-branch-prot-f81` | TKF92 branch transducer (5-state canonical WFST): protein + F81 substitutions, fragment extension parameter `r`. Per [tkf-mixdom/tkf92-wfst-derivation](https://github.com/ihh/tkf-mixdom) |

The TKF91 (Thorne, Kishino, Felsenstein 1991) and TKF92 (1992) models are continuous-time
Markov models of sequence evolution with insertions, deletions, and substitutions; TKF92 adds
a fragment-extension parameter that clusters indels into runs.
`tkf91-root-dna-jc` is a generator (output only) that produces the root sequence;
the `*-branch-*` presets are conditional WFST transducers that evolve an input
sequence to an output sequence. Together root and branch model pairwise evolution.

### Parameterised CLI generators

Any combination of TKF version, root/branch, alphabet, and substitution model can be
constructed on the fly via the CLI flag `--tkfYY-TTT-AAA-MMM`:

- `YY` is `91`, `92`, or `iid`. `iid` is the **zero-indel-rate degeneration of TKF91**:
  the branch transducer collapses to a single emit state with self-loops, with no
  insert/delete states (every input symbol maps 1:1 to an output symbol). The iid root
  is structurally identical to the TKF91 root (a 2-state geometric-length emitter), but
  the length-extension probability is exposed as a free parameter `pExtend` instead of
  being derived from `insRate/delRate` — useful when you want to fix the length
  distribution independently of any indel model. (TKF92 adds the fragment-extension
  parameter `r`.)
- `TTT` is `root` or `branch`.
- `AAA` is `dna`, `rna`, `prot`, `binary`, `unary`, or `custom` (followed by the alphabet string).
- `MMM` is one of:
  - `jc` (Jukes-Cantor; uniform π).
  - `f81` (free per-symbol π_X).
  - `k80` (transition/transversion ratio `tsRatio`, DNA/RNA only).
  - `hky85` (free π_X + `tsRatio`, DNA/RNA only).
  - `id` (no substitution; required for unary alphabet root).
  - `telegraph` (binary 2-state CTMC with independent rates `rate01` and `rate10`;
    stationary π_0 = `rate10/(rate01+rate10)`, π_1 = `rate01/(rate01+rate10)` — these
    are derived via defs from the rates, not free parameters; binary alphabet only).
  - `bsc` (Binary Symmetric Channel: symmetric Telegraph with single rate `flipRate`
    and uniform π = ½; binary alphabet only).
  - `erasure` (Binary Erasure Channel: 0-absorbing Telegraph with rate `eraseRate` —
    only the (1→1) and (1→0) transitions are emitted; binary alphabet only; not valid
    as a root preset because the equilibrium π = (1, 0) is degenerate).

```bash
boss --tkf91-root-dna-jc                   # same as --preset tkf91-root-dna-jc
boss --tkf91-branch-prot-f81               # TKF91 indel structure + F81 protein substitution
boss --tkf92-branch-dna-hky85              # TKF92 fork triad + HKY85
boss --tkf91-branch-custom-jc 0123         # TKF91 + JC over 4-symbol custom alphabet
boss --iid-branch-binary-bsc               # zero-indel-rate BSC: input/output binary 1:1
boss --iid-branch-binary-telegraph         # iid + asymmetric 2-state CTMC on {0, 1}
boss --iid-branch-binary-erasure           # iid + 0-absorbing erasure channel
boss --iid-root-binary-bsc                 # geometric binary emitter (free pExtend) + BSC π
```

The closed-form formulas for K80 and HKY85 transition probabilities are encoded directly in
the resulting WFST's parameter definitions; no numerical eigendecomposition is required at
inference time. The 2×2 closed-form for Telegraph / BSC / Erasure is similarly inlined
(`P_ij(t) = π_j + (δ_ij − π_j) exp(−λt)` with λ = `rate01 + rate10` for Telegraph,
λ = 2`flipRate` for BSC, and the absorbing form for Erasure).

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
