"""DNA pair simulator with homopolymer-dependent error rates.

Simulates ancestor-descendant DNA sequence pairs under a simple
positional model: Jukes-Cantor substitution + insertion + deletion,
with elevated rates in homopolymer runs (>=3 identical bases).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

BASES = ["A", "C", "G", "T"]


def _find_homopolymer_mask(seq_indices: jnp.ndarray, min_run: int = 3) -> jnp.ndarray:
    """Return a boolean mask marking positions in homopolymer runs >= min_run."""
    L = seq_indices.shape[0]
    if L < min_run:
        return jnp.zeros(L, dtype=bool)
    # For each position, count how many consecutive identical bases
    # extend left and right (capped at min_run to avoid full scan).
    mask = jnp.zeros(L, dtype=bool)
    for offset in range(1, min_run):
        same_left = jnp.concatenate([
            jnp.zeros(offset, dtype=bool),
            seq_indices[offset:] == seq_indices[:-offset]
        ])
        same_right = jnp.concatenate([
            seq_indices[:-offset] == seq_indices[offset:],
            jnp.zeros(offset, dtype=bool)
        ])
        mask = mask | same_left | same_right
    # Require that the minimum run length is actually achieved:
    # a position is in a homopolymer if all offsets 1..min_run-1
    # either left or right are identical to it.
    run_left = jnp.ones(L, dtype=bool)
    for offset in range(1, min_run):
        same = jnp.concatenate([
            jnp.zeros(offset, dtype=bool),
            seq_indices[offset:] == seq_indices[:-offset]
        ])
        run_left = run_left & same

    run_right = jnp.ones(L, dtype=bool)
    for offset in range(1, min_run):
        same = jnp.concatenate([
            seq_indices[:-offset] == seq_indices[offset:],
            jnp.zeros(offset, dtype=bool)
        ])
        run_right = run_right & same

    # A position is in a homopolymer run if it starts or ends one
    is_in_run = run_left | run_right
    # Also mark interior positions: propagate the run markers
    for offset in range(1, min_run):
        shifted_left = jnp.concatenate([
            jnp.zeros(offset, dtype=bool), is_in_run[:-offset]
        ])
        shifted_right = jnp.concatenate([
            is_in_run[offset:], jnp.zeros(offset, dtype=bool)
        ])
        is_in_run = is_in_run | (shifted_left & (
            seq_indices == jnp.concatenate([
                jnp.zeros(offset, dtype=jnp.int32), seq_indices[:-offset]
            ])
        )) | (shifted_right & (
            seq_indices == jnp.concatenate([
                seq_indices[offset:], jnp.zeros(offset, dtype=jnp.int32)
            ])
        ))
    return is_in_run


def simulate_dna_pair(
    rng,
    length: int = 100,
    base_sub_rate: float = 0.05,
    base_indel_rate: float = 0.02,
    hp_multiplier: float = 3.0,
    min_hp_run: int = 3,
):
    """Simulate an ancestor-descendant DNA pair.

    Args:
        rng: JAX PRNGKey.
        length: Length of the ancestor sequence.
        base_sub_rate: Per-position substitution probability (baseline).
        base_indel_rate: Per-position insertion/deletion probability (baseline).
        hp_multiplier: Multiplier for rates in homopolymer regions.
        min_hp_run: Minimum run length to count as homopolymer.

    Returns:
        (ancestor, descendant): tuple of strings (DNA sequences).
    """
    k1, k2, k3, k4, k5 = jr.split(rng, 5)

    # Random ancestor
    anc_idx = jr.randint(k1, (length,), 0, 4)
    ancestor = "".join(BASES[int(i)] for i in anc_idx)

    # Per-position rates
    hp_mask = _find_homopolymer_mask(anc_idx, min_hp_run)
    multiplier = jnp.where(hp_mask, hp_multiplier, 1.0)
    sub_rates = jnp.clip(base_sub_rate * multiplier, 0.0, 0.9)
    indel_rates = jnp.clip(base_indel_rate * multiplier, 0.0, 0.4)

    # Apply mutations position by position
    desc_bases = []
    rng_pos = k2
    for i in range(length):
        rng_pos, k_del, k_sub, k_new, k_ins, k_ins_base = jr.split(rng_pos, 6)

        # Deletion?
        if float(jr.uniform(k_del)) < float(indel_rates[i]):
            pass  # skip this position (deletion)
        else:
            # Substitution?
            if float(jr.uniform(k_sub)) < float(sub_rates[i]):
                # JC: pick uniformly from the 3 other bases
                other = [(int(anc_idx[i]) + d) % 4 for d in range(1, 4)]
                choice = int(jr.randint(k_new, (), 0, 3))
                desc_bases.append(BASES[other[choice]])
            else:
                desc_bases.append(BASES[int(anc_idx[i])])

        # Insertion after this position?
        if float(jr.uniform(k_ins)) < float(indel_rates[i]):
            ins_base = int(jr.randint(k_ins_base, (), 0, 4))
            desc_bases.append(BASES[ins_base])

    descendant = "".join(desc_bases)
    return ancestor, descendant


def simulate_batch(rng, n_pairs: int, **kwargs):
    """Simulate a batch of ancestor-descendant DNA pairs.

    Args:
        rng: JAX PRNGKey.
        n_pairs: Number of pairs to simulate.
        **kwargs: Passed to simulate_dna_pair.

    Returns:
        List of (ancestor, descendant) tuples.
    """
    keys = jr.split(rng, n_pairs)
    return [simulate_dna_pair(k, **kwargs) for k in keys]
