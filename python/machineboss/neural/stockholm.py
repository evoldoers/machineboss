"""Minimal Stockholm format parser for multiple sequence alignments.

Parses interleaved Stockholm 1.0 format (.sto files) as used by Pfam.
Supports multi-block interleaved format and #=GF/GC/GS/GR markup (ignored).
"""

from __future__ import annotations

from dataclasses import dataclass
import re

# Standard amino acid alphabet (no gap, no ambiguity)
AA_ALPHA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHA)}
GAP_IDX = 20  # index for gap characters (-, .)


@dataclass
class StockholmMSA:
    """A parsed Stockholm multiple sequence alignment."""
    id: str | None
    names: list[str]
    aligned_seqs: list[str]

    @property
    def n_seqs(self) -> int:
        return len(self.names)

    @property
    def alignment_length(self) -> int:
        if not self.aligned_seqs:
            return 0
        return len(self.aligned_seqs[0])

    def to_onehot(self):
        """Convert to (N, L, 21) one-hot array (20 AA + gap).

        Gap characters (-, .) map to index 20.
        Unknown/ambiguous characters (X, B, Z, etc.) map to uniform over 20 AA.
        Lowercase inserts are treated the same as uppercase.
        """
        import jax.numpy as jnp

        N = self.n_seqs
        L = self.alignment_length
        arr = jnp.zeros((N, L, 21), dtype=jnp.float32)

        for i, seq in enumerate(self.aligned_seqs):
            for j, ch in enumerate(seq):
                ch_upper = ch.upper()
                if ch in "-." or ch_upper == "-":
                    arr = arr.at[i, j, GAP_IDX].set(1.0)
                elif ch_upper in AA_TO_IDX:
                    arr = arr.at[i, j, AA_TO_IDX[ch_upper]].set(1.0)
                else:
                    # Ambiguous: uniform over 20 AA
                    arr = arr.at[i, j, :20].set(1.0 / 20)
        return arr

    def pick_pair(self, i: int, j: int) -> tuple[str, str]:
        """Extract an ungapped sequence pair from rows i and j.

        Columns where both sequences have gaps are removed.
        Returns (seq_i, seq_j) with gap characters preserved
        for columns where one sequence has a gap.
        """
        seq_i = self.aligned_seqs[i]
        seq_j = self.aligned_seqs[j]
        chars_i, chars_j = [], []
        for ci, cj in zip(seq_i, seq_j):
            if ci in "-." and cj in "-.":
                continue  # skip double-gap columns
            chars_i.append(ci)
            chars_j.append(cj)
        return "".join(chars_i), "".join(chars_j)

    def ungapped_pair(self, i: int, j: int) -> tuple[str, str]:
        """Extract a fully ungapped pair (no gap characters).

        Strips all gap characters from both sequences independently.
        Useful for feeding to pairwise alignment.
        """
        seq_i = self.aligned_seqs[i]
        seq_j = self.aligned_seqs[j]
        s_i = re.sub(r"[-.]", "", seq_i).upper()
        s_j = re.sub(r"[-.]", "", seq_j).upper()
        return s_i, s_j


def parse_stockholm(text: str) -> StockholmMSA:
    """Parse a Stockholm 1.0 format alignment from a string.

    Args:
        text: Full content of a .sto file.

    Returns:
        StockholmMSA with parsed names and aligned sequences.
    """
    lines = text.strip().split("\n")

    # Verify header
    if not lines or not lines[0].startswith("# STOCKHOLM 1.0"):
        raise ValueError("Not a Stockholm 1.0 format file")

    msa_id = None
    name_order: list[str] = []
    seq_blocks: dict[str, list[str]] = {}

    for line in lines[1:]:
        line = line.rstrip()

        if line == "//" or not line:
            continue

        # Metadata
        if line.startswith("#=GF"):
            parts = line.split(None, 2)
            if len(parts) >= 3 and parts[1] == "ID":
                msa_id = parts[2]
            continue
        if line.startswith("#"):
            continue

        # Sequence line: name sequence
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue

        name, seq_fragment = parts
        if name not in seq_blocks:
            name_order.append(name)
            seq_blocks[name] = []
        seq_blocks[name].append(seq_fragment)

    # Concatenate blocks
    aligned_seqs = ["".join(seq_blocks[name]) for name in name_order]

    # Validate lengths
    if aligned_seqs:
        L = len(aligned_seqs[0])
        for i, seq in enumerate(aligned_seqs):
            if len(seq) != L:
                raise ValueError(
                    f"Sequence {name_order[i]} has length {len(seq)}, "
                    f"expected {L}")

    return StockholmMSA(id=msa_id, names=name_order, aligned_seqs=aligned_seqs)


def parse_stockholm_file(path: str) -> StockholmMSA:
    """Parse a Stockholm file from a path."""
    with open(path) as f:
        return parse_stockholm(f.read())
