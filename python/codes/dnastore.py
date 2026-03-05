#!/usr/bin/env python3
"""Generate a DNA encoding transducer that avoids specified motifs.

Builds a filtered De Bruijn graph over DNA k-mers, excluding k-mers that
contain forbidden motifs or homopolymer runs. The resulting transducer reads
radix-annotated input symbols (matching mixradar.py output format, e.g.
"0_3", "1_3", "2_3" at positions with 3 valid successor bases) and outputs
DNA bases.

At positions where only one successor base is valid (radix 1), the transducer
emits the forced base without consuming input. At positions where all four
bases are valid (radix 4), four input symbols are accepted.

Usage:
    python dnastore.py CONTEXT_LEN [--motif MOTIF ...] [--json] [--dot] [--stats]

Examples:
    # 4-mer context, avoid homopolymers only (Goldman-style ternary code)
    python dnastore.py 4 --json

    # 4-mer context, also avoid AfeI restriction site
    python dnastore.py 4 --motif AGCGCT --json

    # Compose with mixradar for full binary-to-DNA encoding
    python dnastore.py 4 --motif AGCGCT --json | \\
      boss - --input-chars 10110011 --beam-encode
"""

import argparse
import json
import sys
from collections import deque

BASES = "ACGT"
BASE2IDX = {b: i for i, b in enumerate(BASES)}


def kmer_to_str(kmer, length):
    """Convert integer k-mer to DNA string."""
    s = []
    for _ in range(length):
        s.append(BASES[kmer & 3])
        kmer >>= 2
    return ''.join(reversed(s))


def str_to_kmer(s):
    """Convert DNA string to integer k-mer."""
    k = 0
    for c in s:
        k = (k << 2) | BASE2IDX[c]
    return k


def reverse_complement(s):
    """Return reverse complement of DNA string."""
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(comp[c] for c in reversed(s))


def has_homopolymer(s, max_run=1):
    """Check if string has a homopolymer run longer than max_run."""
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
            if count > max_run:
                return True
        else:
            count = 1
    return False


def contains_motif(s, motifs):
    """Check if string contains any forbidden motif as a substring."""
    for motif in motifs:
        if motif in s:
            return True
    return False


def build_graph(context_len, motifs, no_homopolymer=True, avoid_rc=False):
    """Build filtered De Bruijn graph over DNA k-mers.

    Returns (valid_kmers, edges) where edges maps kmer -> [(base_idx, succ_kmer)].
    """
    n_kmers = 4 ** context_len
    mask = n_kmers - 1

    all_motifs = list(motifs)
    if avoid_rc:
        for m in motifs:
            rc = reverse_complement(m)
            if rc not in all_motifs:
                all_motifs.append(rc)

    # Filter valid k-mers
    valid = set()
    for k in range(n_kmers):
        s = kmer_to_str(k, context_len)
        if no_homopolymer and has_homopolymer(s):
            continue
        if contains_motif(s, all_motifs):
            continue
        valid.add(k)

    # Build edges
    edges = {}
    for k in valid:
        k_str = kmer_to_str(k, context_len)
        successors = []
        for b in range(4):
            succ = ((k << 2) | b) & mask
            if succ not in valid:
                continue
            # Check that extending by this base doesn't create a forbidden motif
            # at the boundary (motifs that span old context + new base)
            extended = k_str + BASES[b]
            if contains_motif(extended, all_motifs):
                continue
            if no_homopolymer and k_str[-1] == BASES[b]:
                continue
            successors.append((b, succ))
        edges[k] = successors

    # Prune dead ends iteratively
    changed = True
    while changed:
        changed = False
        has_incoming = set()
        for k in valid:
            for _, succ in edges.get(k, []):
                has_incoming.add(succ)
        to_remove = set()
        for k in list(valid):
            if not edges.get(k):
                to_remove.add(k)
            elif k not in has_incoming:
                to_remove.add(k)
        if to_remove:
            valid -= to_remove
            for k in to_remove:
                edges.pop(k, None)
            for k in list(valid):
                edges[k] = [(b, s) for b, s in edges[k] if s in valid]
            changed = True

    return valid, edges


def build_transducer(context_len, motifs, no_homopolymer=True, avoid_rc=False):
    """Build DNA encoding transducer as Machine Boss JSON.

    States are k-mers in a filtered De Bruijn graph.
    Input alphabet: radix-annotated symbols "digit_radix" (matching mixradar.py).
    Output alphabet: DNA bases {A, C, G, T}.
    """
    valid, edges = build_graph(context_len, motifs, no_homopolymer, avoid_rc)

    if not valid:
        print("Error: no valid k-mers remain after filtering. "
              "Try a shorter context length or fewer motifs.", file=sys.stderr)
        sys.exit(1)

    sorted_kmers = sorted(valid)
    # State indices: 0 = start, 1..N = k-mer states, N+1 = end
    kmer_to_state = {k: i + 1 for i, k in enumerate(sorted_kmers)}
    end_state = len(sorted_kmers) + 1

    states = []

    # Start state: silent transitions to all valid k-mers
    states.append({
        "id": "start",
        "trans": [{"to": kmer_to_state[k]} for k in sorted_kmers],
    })

    # K-mer states
    for k in sorted_kmers:
        succ = edges[k]
        radix = len(succ)
        k_str = kmer_to_str(k, context_len)

        trans = []
        if radix == 1:
            # Forced base: no input consumed, emit the only valid base
            b, s = succ[0]
            trans.append({"to": kmer_to_state[s], "out": BASES[b]})
        else:
            # Radix R: accept digit_radix input symbols
            for i, (b, s) in enumerate(succ):
                trans.append({
                    "to": kmer_to_state[s],
                    "in": f"{i}_{radix}",
                    "out": BASES[b],
                })
        # Silent transition to end state
        trans.append({"to": end_state})

        states.append({"id": k_str, "trans": trans})

    # End state
    states.append({"id": "end"})

    return {"state": states}


def radix_stats(context_len, motifs, no_homopolymer=True, avoid_rc=False):
    """Compute statistics about the radix distribution."""
    valid, edges = build_graph(context_len, motifs, no_homopolymer, avoid_rc)
    if not valid:
        return "No valid k-mers."

    radix_counts = {}
    for k in sorted(valid):
        r = len(edges[k])
        radix_counts[r] = radix_counts.get(r, 0) + 1

    lines = []
    lines.append(f"Context length: {context_len}")
    lines.append(f"Valid k-mers: {len(valid)} / {4 ** context_len}")
    lines.append(f"Motifs avoided: {motifs if motifs else '(none)'}")
    lines.append(f"Homopolymer avoidance: {no_homopolymer}")
    lines.append("")
    lines.append("Radix distribution (radix: count):")
    total_bits = 0
    total_states = 0
    for r in sorted(radix_counts):
        n = radix_counts[r]
        lines.append(f"  radix {r}: {n} states")
        if r >= 2:
            import math
            total_bits += n * math.log2(r)
            total_states += n
    if total_states > 0:
        lines.append(f"\nMean bits/position (encoding states only): "
                     f"{total_bits / total_states:.3f}")
        lines.append(f"Mean bits/position (all states): "
                     f"{total_bits / len(valid):.3f}")
    return '\n'.join(lines)


def to_dot(context_len, motifs, no_homopolymer=True, avoid_rc=False):
    """Generate GraphViz DOT representation."""
    valid, edges = build_graph(context_len, motifs, no_homopolymer, avoid_rc)
    lines = ["digraph dnastore {", "  rankdir=LR;"]
    for k in sorted(valid):
        k_str = kmer_to_str(k, context_len)
        radix = len(edges[k])
        lines.append(f'  "{k_str}" [label="{k_str}\\n(r={radix})"];')
        for i, (b, s) in enumerate(edges[k]):
            s_str = kmer_to_str(s, context_len)
            if radix == 1:
                label = BASES[b]
            else:
                label = f"{i}_{radix}:{BASES[b]}"
            lines.append(f'  "{k_str}" -> "{s_str}" [label="{label}"];')
    lines.append("}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate a DNA encoding transducer avoiding specified motifs')
    parser.add_argument('context_len', type=int,
                        help='Context length (k-mer size)')
    parser.add_argument('--motif', action='append', default=[],
                        help='Motif to avoid (may be repeated)')
    parser.add_argument('--avoid-rc', action='store_true',
                        help='Also avoid reverse complements of motifs')
    parser.add_argument('--allow-homopolymers', action='store_true',
                        help='Allow homopolymer runs (default: forbidden)')
    parser.add_argument('-j', '--json', action='store_true',
                        help='Output Machine Boss JSON')
    parser.add_argument('-d', '--dot', action='store_true',
                        help='Output GraphViz DOT')
    parser.add_argument('-s', '--stats', action='store_true',
                        help='Print statistics')

    args = parser.parse_args()

    no_hp = not args.allow_homopolymers

    if args.stats:
        print(radix_stats(args.context_len, args.motif, no_hp, args.avoid_rc))
    elif args.dot:
        print(to_dot(args.context_len, args.motif, no_hp, args.avoid_rc))
    elif args.json:
        machine = build_transducer(
            args.context_len, args.motif, no_hp, args.avoid_rc)
        json.dump(machine, sys.stdout, indent=2)
        print()
    else:
        print(radix_stats(args.context_len, args.motif, no_hp, args.avoid_rc))


if __name__ == '__main__':
    main()
