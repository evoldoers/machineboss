#!/usr/bin/env python
"""Train a neural TKF92 protein transducer on a Stockholm MSA.

An MSA transformer reads a Pfam-style alignment; parameter heads predict
per-position (t, insRate, delRate, r, pi_0..pi_19) for a 7-state TKF92
WFST. Gradients flow from the Forward log-likelihood back through the
DP into the transformer.

Usage:
    python examples/train_neural_tkf92.py --sto alignment.sto [--steps 50]
    python examples/train_neural_tkf92.py --help
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from machineboss.jax.jax_weight import ParameterizedMachine
from machineboss.jax.dp_neural import neural_log_forward_tok
from machineboss.neural.tkf92 import (
    make_tkf92_machine, TKF92Heads, heads_to_dp_params, AA_ALPHA,
)
from machineboss.neural.msa_transformer import MSATransformer
from machineboss.neural.stockholm import parse_stockholm_file


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sto", required=True, help="Stockholm alignment file")
    parser.add_argument("--steps", type=int, default=50, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=64,
                        help="Transformer model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="Transformer layers")
    parser.add_argument("--pairs-per-step", type=int, default=1,
                        help="Sequence pairs to sample per step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = jr.PRNGKey(args.seed)

    # Parse MSA
    msa = parse_stockholm_file(args.sto)
    print(f"MSA: {msa.id or 'unnamed'}, {msa.n_seqs} sequences, "
          f"length {msa.alignment_length}")
    msa_onehot = msa.to_onehot()  # (N, L, 21)

    # Build machine
    machine = make_tkf92_machine()
    pm = ParameterizedMachine.from_machine(machine)
    print(f"Machine: {pm.n_states} states, {len(pm.free_params)} free params")

    # Build tokenizer
    aa_map = {aa: i for i, aa in enumerate(pm.input_tokens)}

    def tokenize_aa(seq):
        return jnp.array([aa_map.get(c, 0) for c in seq], dtype=jnp.int32)

    # Initialize models
    transformer = MSATransformer(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers)
    heads = TKF92Heads()

    rng, k1, k2 = jr.split(rng, 3)
    t_params = transformer.init(k1, msa_onehot)
    dummy_emb = transformer.apply(t_params, msa_onehot)
    h_params = heads.init(k2, dummy_emb)

    all_params = {"transformer": t_params, "heads": h_params}
    n_model_params = sum(p.size for p in jax.tree.leaves(all_params))
    print(f"Model parameters: {n_model_params}")

    # Optimizer
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(all_params)

    # Loss function
    def loss_fn(all_params, in_tok, out_tok):
        emb = transformer.apply(all_params["transformer"], msa_onehot)
        t, insRate, delRate, r, pi = heads.apply(all_params["heads"], emb)
        dp_params = heads_to_dp_params(t, insRate, delRate, r, pi)
        ll = neural_log_forward_tok(pm, in_tok, out_tok, dp_params)
        return -ll

    # Training loop
    print(f"\nTraining for {args.steps} steps...")
    for step in range(args.steps):
        rng, k_pair = jr.split(rng)

        # Sample a random pair
        idxs = jr.randint(k_pair, (2,), 0, msa.n_seqs)
        i, j = int(idxs[0]), int(idxs[1])
        if i == j:
            j = (j + 1) % msa.n_seqs
        seq_i, seq_j = msa.ungapped_pair(i, j)

        if not seq_i or not seq_j:
            continue

        in_tok = tokenize_aa(seq_i)
        out_tok = tokenize_aa(seq_j)

        loss, grads = jax.value_and_grad(loss_fn)(all_params, in_tok, out_tok)
        updates, opt_state = optimizer.update(grads, opt_state)
        all_params = optax.apply_updates(all_params, updates)

        if step % 5 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss={float(loss):8.3f}  "
                  f"pair=({msa.names[i]}, {msa.names[j]})  "
                  f"|seq|=({len(seq_i)}, {len(seq_j)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
