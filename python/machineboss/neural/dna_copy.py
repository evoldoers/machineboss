"""Neural DNA copy transducer: CNN predicts per-position WFST parameters.

A 6-state TKF91-like machine (begin, wait, match, insert, delete, end)
with Jukes-Cantor substitution. A small 1D CNN maps a one-hot DNA input
sequence to per-position parameters (t, pIns, pDel) that feed into the
ParameterizedMachine + neural_log_forward_tok infrastructure.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..machine import Machine, MachineState, MachineTransition

DNA_BASES = ["A", "C", "G", "T"]


def make_dna_copy_machine() -> Machine:
    """Build a 6-state DNA copy transducer with JC substitution.

    States: begin(0), wait(1), match(2), insert(3), delete(4), end(5)

    Free parameters: t, pIns, pDel
    Defs: pNoSub = exp(-t), pSub = 1-pNoSub, pDiff = pSub/4, pSame = pNoSub+pDiff
    """
    defs = {
        "pNoSub": {"exp": {"*": [-1, "t"]}},
        "pSub": {"not": "pNoSub"},
        "pDiff": {"/": ["pSub", 4]},
        "pSame": {"+": ["pNoSub", "pDiff"]},
    }

    # State 0: begin
    begin = MachineState(name="begin", trans=[
        MachineTransition(dest=3, weight="pIns"),           # -> insert
        MachineTransition(dest=1, weight={"not": "pIns"}),  # -> wait
    ])

    # State 1: wait
    wait = MachineState(name="wait", trans=[
        MachineTransition(dest=2, weight={"not": "pDel"}),  # -> match
        MachineTransition(dest=4, weight="pDel"),            # -> delete
        MachineTransition(dest=5),                           # -> end (weight 1)
    ])

    # State 2: match (4x4 JC transitions back to begin)
    match_trans = []
    for inp in DNA_BASES:
        for out in DNA_BASES:
            w = "pSame" if inp == out else "pDiff"
            match_trans.append(
                MachineTransition(dest=0, weight=w, input=inp, output=out))
    match = MachineState(name="match", trans=match_trans)

    # State 3: insert (emit output, uniform 1/4)
    insert = MachineState(name="insert", trans=[
        MachineTransition(dest=0, weight={"/": [1, 4]}, output=b)
        for b in DNA_BASES
    ])

    # State 4: delete (consume input)
    delete = MachineState(name="delete", trans=[
        MachineTransition(dest=0, input=b) for b in DNA_BASES
    ])

    # State 5: end
    end = MachineState(name="end", trans=[])

    return Machine(
        state=[begin, wait, match, insert, delete, end],
        defs=defs,
    )


def onehot_dna(seq: str) -> jnp.ndarray:
    """Convert a DNA string to (L, 4) one-hot array."""
    base_map = {b: i for i, b in enumerate(DNA_BASES)}
    indices = [base_map[c] for c in seq.upper()]
    return jnp.eye(4, dtype=jnp.float32)[jnp.array(indices)]


def tokenize_dna(seq: str, pm) -> jnp.ndarray:
    """Convert a DNA string to 1-based token indices for a ParameterizedMachine."""
    return jnp.array(pm.tokenize_input(list(seq.upper())), dtype=jnp.int32)


try:
    import flax.linen as nn

    class DNACopyCNN(nn.Module):
        """1D CNN that maps one-hot DNA to per-position (t, pIns, pDel).

        Input:  (L, 4) one-hot DNA
        Output: dict of parameter arrays each (Li+1, 1)
            t:    softplus(raw) + 0.01   (positive evolutionary distance)
            pIns: sigmoid(raw) * 0.3     (insertion probability, capped)
            pDel: sigmoid(raw) * 0.3     (deletion probability, capped)
        """
        hidden: int = 32
        kernel: int = 5

        @nn.compact
        def __call__(self, x):
            # x: (L, 4)
            x = nn.Conv(features=self.hidden, kernel_size=(self.kernel,),
                        padding="SAME")(x)
            x = nn.relu(x)
            x = nn.Conv(features=self.hidden, kernel_size=(self.kernel,),
                        padding="SAME")(x)
            x = nn.relu(x)
            raw = nn.Conv(features=3, kernel_size=(1,), padding="SAME")(x)
            # raw: (L, 3)
            t = jax.nn.softplus(raw[:, 0]) + 0.01
            pIns = jax.nn.sigmoid(raw[:, 1]) * 0.3
            pDel = jax.nn.sigmoid(raw[:, 2]) * 0.3
            return t, pIns, pDel

    def cnn_params_to_dp_params(t, pIns, pDel):
        """Reshape CNN outputs to (Li+1, 1) parameter tensors for DP.

        The DP grid has Li+1 rows (0 = boundary, 1..Li = input positions).
        Boundary row uses the first position's parameters.
        """
        # Prepend boundary (use first position values)
        t_pad = jnp.concatenate([t[:1], t])[:, None]         # (Li+1, 1)
        pIns_pad = jnp.concatenate([pIns[:1], pIns])[:, None]
        pDel_pad = jnp.concatenate([pDel[:1], pDel])[:, None]
        return {"t": t_pad, "pIns": pIns_pad, "pDel": pDel_pad}

    def make_loss_fn(pm):
        """Create a loss function: CNN params -> -mean log-likelihood.

        Args:
            pm: ParameterizedMachine compiled from make_dna_copy_machine().

        Returns:
            loss_fn(cnn_params, model, input_seq_onehot, input_tokens, output_tokens)
        """
        from ..jax.dp_neural import neural_log_forward_tok

        def loss_fn(cnn_params, model, input_onehot, input_tokens, output_tokens):
            t, pIns, pDel = model.apply(cnn_params, input_onehot)
            dp_params = cnn_params_to_dp_params(t, pIns, pDel)
            ll = neural_log_forward_tok(pm, input_tokens, output_tokens, dp_params)
            return -ll

        return loss_fn

except ImportError:
    pass  # Flax not available; machine construction still works
