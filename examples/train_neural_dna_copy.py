#!/usr/bin/env python
"""Train a neural DNA copy transducer on simulated data.

A 1D CNN maps one-hot DNA to per-position parameters (t, pIns, pDel)
that feed into a 6-state TKF91-like WFST. The Forward algorithm computes
log-likelihoods; gradients flow back through the DP into the CNN.

Usage:
    python examples/train_neural_dna_copy.py [--steps 50] [--length 30] [--lr 1e-3]
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from machineboss.jax.jax_weight import ParameterizedMachine
from machineboss.jax.dp_neural import neural_log_forward_tok
from machineboss.neural.dna_copy import (
    make_dna_copy_machine, DNACopyCNN, onehot_dna, tokenize_dna,
    cnn_params_to_dp_params,
)
from machineboss.neural.simulator import simulate_dna_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=50, help="Training steps")
    parser.add_argument("--length", type=int, default=30, help="Ancestor sequence length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=32, help="CNN hidden channels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sub-rate", type=float, default=0.08,
                        help="Base substitution rate for simulator")
    parser.add_argument("--indel-rate", type=float, default=0.03,
                        help="Base indel rate for simulator")
    parser.add_argument("--hp-multiplier", type=float, default=3.0,
                        help="Homopolymer rate multiplier")
    args = parser.parse_args()

    rng = jr.PRNGKey(args.seed)

    # Build machine and compile
    machine = make_dna_copy_machine()
    pm = ParameterizedMachine.from_machine(machine)
    print(f"Machine: {pm.n_states} states, free_params={pm.free_params}")
    print(f"Input alphabet: {pm.input_tokens}")
    print(f"Output alphabet: {pm.output_tokens}")

    # Initialize CNN
    model = DNACopyCNN(hidden=args.hidden)
    dummy = onehot_dna("A" * args.length)
    rng, init_key = jr.split(rng)
    cnn_params = model.init(init_key, dummy)
    n_params = sum(p.size for p in jax.tree.leaves(cnn_params))
    print(f"CNN parameters: {n_params}")

    # Optimizer
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(cnn_params)

    # Loss function
    def loss_fn(cnn_params, input_onehot, input_tokens, output_tokens):
        t, pIns, pDel = model.apply(cnn_params, input_onehot)
        dp_params = cnn_params_to_dp_params(t, pIns, pDel)
        ll = neural_log_forward_tok(pm, input_tokens, output_tokens, dp_params)
        return -ll

    # Training loop
    print(f"\nTraining for {args.steps} steps on simulated pairs (L={args.length})...")
    for step in range(args.steps):
        rng, sim_key = jr.split(rng)
        anc, desc = simulate_dna_pair(
            sim_key, length=args.length,
            base_sub_rate=args.sub_rate,
            base_indel_rate=args.indel_rate,
            hp_multiplier=args.hp_multiplier,
        )

        x = onehot_dna(anc)
        in_tok = tokenize_dna(anc, pm)
        out_tok = tokenize_dna(desc, pm)

        loss, grads = jax.value_and_grad(loss_fn)(cnn_params, x, in_tok, out_tok)
        updates, opt_state = optimizer.update(grads, opt_state)
        cnn_params = optax.apply_updates(cnn_params, updates)

        if step % 5 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss={float(loss):8.3f}  "
                  f"|anc|={len(anc)}  |desc|={len(desc)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
