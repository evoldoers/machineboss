---
title: DNA Storage
nav_order: 12
permalink: /dna-storage/
---

# Tutorial: DNA Data Storage

This tutorial covers using Machine Boss for DNA data storage coding:
encoding binary data as DNA sequences that satisfy biochemical constraints,
and decoding DNA back to binary.
It covers the Goldman et al. (2013) ternary code, related presets,
error-correcting codes, and the mix-radix encoder generator.

## Table of Contents
{: .no_toc }

1. TOC
{:toc}

## Background

Synthetic DNA has been proposed as a medium for long-term digital data storage.
The key challenge is converting binary data into DNA sequences
that avoid problematic features such as homopolymer runs
(repeated identical bases like AAAA),
which cause errors in synthesis and sequencing.

Goldman et al. (2013), in their Nature letter
"Towards practical, high-capacity, low-maintenance information storage in synthesized DNA,"
introduced an elegant two-stage encoding:

1. **Binary to ternary**: Convert the binary bitstream to base-3 (ternary) digits.
2. **Ternary to non-repeating DNA**: Map each ternary digit to one of three DNA bases
   that differs from the previous base, guaranteeing no homopolymer runs.

Machine Boss implements both stages as preset transducers.

## The Goldman et al. Code

### Stage 1: Binary to ternary (`bintern`)

The `bintern` preset converts blocks of 5 binary bits to 2 ternary digits.
This achieves a rate of 5/4 = 1.25 bits per ternary digit,
close to the theoretical maximum of log₂(3) ≈ 1.585 bits per trit.

```bash
boss --preset bintern
```

The encoding uses Huffman-like block coding:
each 5-bit input block (with a possible end-of-file marker)
maps to a pair of ternary digits from the alphabet {0, 1, 2}.

To encode a binary string:

```bash
boss --preset bintern --input-chars 1010101 --beam-encode
```

This outputs a ternary string like `12022212`.

### Stage 2: Ternary to non-repeating DNA (`terndna`)

The `terndna` preset converts a ternary string to DNA
using a rotating alphabet scheme.
The machine tracks the previously emitted base and, for each ternary digit,
selects from the three bases that differ from it:

| Previous base | Trit 0 | Trit 1 | Trit 2 |
|---------------|--------|--------|--------|
| A             | C      | G      | T      |
| C             | A      | G      | T      |
| G             | A      | C      | T      |
| T             | A      | C      | G      |

This guarantees that consecutive bases always differ — no homopolymer runs.

```bash
boss --preset terndna --input-chars 12022212 --beam-encode
```

### The composite pipeline

Both stages can be composed on a single command line:

```bash
boss --preset bintern --preset terndna --input-chars 1010101 --beam-encode
```

This outputs the DNA sequence directly (e.g. `CGATATGC`).

### Decoding

To decode, supply the DNA as the output and decode back to binary:

```bash
boss --preset bintern --preset terndna --output-chars CGATATGC --beam-decode
```

This recovers the original binary string `1010101`.

Machine Boss uses beam search for both encoding and decoding.
Beam search is generally much faster than the alternative prefix search (`--prefix-decode`),
though prefix search is exact.

## Related Presets

### `AfeI`: Restriction enzyme avoidance

The `AfeI` preset works like `terndna` but additionally avoids
the AfeI restriction enzyme recognition site AGCGCT.
It tracks enough context (the last several bases) to detect
and prevent this hexamer from appearing in the output:

```bash
boss --preset bintern --preset AfeI --input-chars 1010101 --beam-encode
```

This is useful when the synthesized DNA will be processed with restriction enzymes.

### `nontern`: Pre-composed ternary encoding

The `nontern` preset is the precomputed composition of `bintern` and `terndna`.
It maps bit blocks directly to non-repeating ternary-DNA tokens in a single transducer:

```bash
boss --preset nontern --input-chars 1010101 --beam-encode
```

### `bitbase`: Direct binary-to-DNA

The `bitbase` preset maps each binary bit directly to a DNA base,
choosing from 3 alternatives that differ from the previous base.
This achieves a rate of 1 bit per base (vs. the theoretical maximum of ~1.585):

```bash
boss --preset bitbase --input-chars 1010101 --beam-encode
```

### `bytern`: Byte-oriented encoding

The `bytern` preset provides byte-oriented binary-to-ternary conversion,
mapping 8-bit input blocks to 3 pairs of ternary digits:

```bash
boss --preset bytern --input-chars 10110011 --beam-encode
```

### `bitcod`: Simple padding code

The `bitcod` preset is a trivial binary-to-ternary code
that maps each bit `b` to the ternary string `b12`.
It has a rate of only 1/3 bit per trit and is mainly useful for testing.

## Error-Correcting Codes

Machine Boss also provides error-correcting codes as transducers,
which can be composed with the DNA storage pipeline
to add redundancy for error recovery.

### Repetition code (`hamming31`)

The simplest error-correcting code: each input bit is repeated 3 times.

```bash
boss --preset hamming31 --input-chars 101 --beam-encode
# Outputs: 111000111
```

### Hamming(7,4) code (`hamming74`)

The Hamming(7,4) code encodes 4 data bits into a 7-bit codeword
with 3 parity bits, allowing single-bit error correction:

```bash
boss --preset hamming74 --input-chars 1011 --beam-encode
```

### Composing with the DNA pipeline

Error-correcting codes can be composed with the DNA storage presets:

```bash
# Encode: binary → Hamming(7,4) → ternary → non-repeating DNA
boss --preset hamming74 --preset bintern --preset terndna \
     --input-chars 1011 --beam-encode
```

Decoding reverses the pipeline, with the beam search decoder
implicitly finding the most likely codeword.

## Code Generation with Python

The `python/codes/` directory contains Python scripts for generating
custom coding transducers.

### Hamming code generator

The `hamming74.py` script generates a Hamming(7,4) code transducer:

```bash
python python/codes/hamming74.py --json >my_hamming.json
python python/codes/hamming74.py --dot | dot -Tpdf >hamming.pdf
```

### Mix-radix encoder generator

The `mixradar.py` script generates variable-length mix-radix codes —
a generalization of arithmetic coding that uses mixed radices (2, 3, and 4)
instead of a fixed base.

**Why mixed radices?**
The Goldman ternary code assumes that at every position
you have exactly 3 choices (the bases that differ from the previous one).
But in practice, a DNA storage scheme may need to avoid
more than just homopolymers.
For example, you might want to exclude restriction enzyme recognition sites
(like AfeI's AGCGCT) so the stored DNA survives enzymatic processing.
Other constraints could include avoiding long GC-rich or AT-rich stretches,
secondary structure motifs, or sequences that interfere with synthesis.

These context-dependent constraints mean that at some positions
you may have fewer than 3 bases available.
If the previous two bases were AG and you need to avoid AGC,
you can only choose from {A, G, T} — but G is already excluded
by the homopolymer rule, leaving just {A, T}: only 2 options,
so you can encode at most 1 bit at that position.
Sometimes the constraints leave only a single valid base,
in which case you can encode no information at all —
you simply emit the forced base and wait for the next position
to offer more choices.

A fixed-radix code cannot handle this gracefully:
it assumes a constant number of choices per position.
A mix-radix code adapts naturally,
using radix 4 when all bases are available,
radix 3 when one is excluded (the common case),
radix 2 when two are excluded,
and radix 1 (a forced emit) when only one base is possible.
This extracts the maximum information from every position
regardless of how many options the constraints leave open.

The idea is to encode a block of binary input bits (plus an end-of-file symbol)
into a variable-length sequence of digits drawn from alphabets of different sizes.
The encoder assigns probability intervals to input words (based on their binary probabilities
and an EOF probability), then subdivides each interval using the largest radix
that fits, greedily emitting digits.

```bash
# Generate a mix-radix encoder for 5-bit blocks
python python/codes/mixradar.py 5 --json >mixradar5.json

# Visualize the transducer
python python/codes/mixradar.py 5 --dot | dot -Tpdf >mixradar5.pdf

# Show encoding statistics
python python/codes/mixradar.py 5 --stats
```

The `--maxradix` flag controls the maximum alphabet size (default 4, for DNA).
With `--maxradix 3` you get a pure ternary code comparable to `bintern`.

The script works by:
1. Building a prefix tree of all binary input words of the given block length, plus EOF.
2. Assigning probability intervals to each word
   (uniform bits with probability 1/2, EOF with configurable probability).
3. Subdividing intervals by the largest radix (2, 3, or 4) whose digit boundaries
   fall cleanly within the interval.
4. Emitting output digits as transitions in a transducer.
5. Pruning and merging equivalent states to minimize the machine.

This produces a near-optimal variable-length code that can be composed with `terndna` or `AfeI`
for DNA storage:

```bash
python python/codes/mixradar.py 8 --json | \
  boss - --preset terndna --input-chars 10110011 --beam-encode
```

## References

- Goldman, N., Bertone, P., Chen, S., Dessimoz, C., LeProust, E. M., Sipos, B., & Birney, E. (2013).
  Towards practical, high-capacity, low-maintenance information storage in synthesized DNA.
  *Nature*, 494(7435), 77–80.
  [doi:10.1038/nature11875](https://doi.org/10.1038/nature11875)
