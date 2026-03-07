"""Neural TKF92 protein transducer: MSA transformer → per-position WFST parameters.

A 7-state TKF92 machine (begin, orphan, wait, match, insert, delete, end)
with F81 amino acid substitution and fragment extension parameter r.
An MSA transformer produces per-position parameters via parameter heads.

The TKF92 model extends TKF91 with a fragment extension probability r:
each match/insert/delete state has a self-loop with probability r,
allowing runs of consecutive matches, inserts, or deletes within a
single "fragment". On termination (probability 1-r), standard TKF91
transitions apply.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..machine import Machine, MachineState, MachineTransition

# 20 standard amino acids (alphabetical)
AA_ALPHA = sorted("ACDEFGHIKLMNPQRSTVWY")
N_AA = 20


def make_tkf92_machine() -> Machine:
    """Build a 7-state TKF92 protein transducer with F81 substitution.

    States: begin(0), orphan(1), wait(2), match(3), insert(4), delete(5), end(6)

    Free parameters:
        t         - evolutionary time
        insRate   - insertion rate (lambda)
        delRate   - deletion rate (mu), must be > insRate
        r         - fragment extension probability (0 < r < 1)
        pi_0..pi_19 - amino acid equilibrium frequencies

    Defs (same as tkf91branch.json + F81 + fragment extension):
        pNoDeletion, pDeletion, pNoInsertion, pInsertion,
        delInsRatio, pDescendants, pNoDescendants, pOrphans, pNoOrphans
        pNoSub, pSub

    Substitution weights use F81 model:
        P(j|i, t) = pNoSub * delta(i,j) + pSub * pi_j
    """
    pi_names = [f"pi_{i}" for i in range(N_AA)]

    defs = {
        "pNoDeletion": {"exp": {"*": [-1, {"*": ["delRate", "t"]}]}},
        "pDeletion": {"not": "pNoDeletion"},
        "pNoInsertion": {"exp": {"*": [-1, {"*": ["insRate", "t"]}]}},
        "pInsertion": {"not": "pNoInsertion"},
        "delInsRatio": {"/": ["pNoDeletion", "pNoInsertion"]},
        "pDescendants": {"/": [
            {"*": ["insRate", {"not": "delInsRatio"}]},
            {"-": ["delRate", {"*": ["insRate", "delInsRatio"]}]}
        ]},
        "pNoDescendants": {"not": "pDescendants"},
        "pOrphans": {"not": {"*": [
            {"/": ["delRate", "insRate"]},
            {"/": ["pDescendants", "pDeletion"]}
        ]}},
        "pNoOrphans": {"not": "pOrphans"},
        "pNoSub": {"exp": {"*": [-1, "t"]}},
        "pSub": {"not": "pNoSub"},
    }

    # Helper: TKF91 transition weight × (1 - r) for non-self-loop transitions
    def scaled(w):
        """Weight w multiplied by (1 - r)."""
        return {"*": [{"not": "r"}, w]}

    # Helper: F81 substitution weight for amino acid pair (i -> j)
    def f81_weight(j_idx):
        """F81: pNoSub * delta(i,j) + pSub * pi_j  →  pSame_j or pDiff_j."""
        pj = pi_names[j_idx]
        return {"+": ["pNoSub", {"*": ["pSub", pj]}]}

    def f81_diff(j_idx):
        """F81 for i != j: pSub * pi_j."""
        return {"*": ["pSub", pi_names[j_idx]]}

    # State 0: begin
    begin = MachineState(name="begin", trans=[
        MachineTransition(dest=4, weight="pDescendants"),     # -> insert
        MachineTransition(dest=2, weight="pNoDescendants"),   # -> wait
    ])

    # State 1: orphan
    orphan = MachineState(name="orphan", trans=[
        MachineTransition(dest=4, weight="pOrphans"),         # -> insert
        MachineTransition(dest=2, weight="pNoOrphans"),       # -> wait
    ])

    # State 2: wait
    wait = MachineState(name="wait", trans=[
        MachineTransition(dest=3, weight={"not": "pDel"}),    # -> match
        MachineTransition(dest=5, weight="pDel"),              # -> delete
        MachineTransition(dest=6),                             # -> end (weight 1)
    ])

    # Note: wait uses pDel as a convenience alias
    defs["pDel"] = "pDeletion"

    # State 3: match
    # Self-loop (fragment extension): match -> match with weight r × F81
    # Exit transitions: match -> begin with weight (1-r) × TKF91 weight × F81
    match_trans = []
    for i_idx, inp in enumerate(AA_ALPHA):
        for j_idx, out in enumerate(AA_ALPHA):
            if inp == out:
                w_sub = f81_weight(j_idx)
            else:
                w_sub = f81_diff(j_idx)
            # Self-loop: fragment extension
            match_trans.append(MachineTransition(
                dest=3, weight={"*": ["r", w_sub]},
                input=inp, output=out))
            # Exit to begin: (1-r) × substitution weight
            match_trans.append(MachineTransition(
                dest=0, weight=scaled(w_sub),
                input=inp, output=out))
    match = MachineState(name="match", trans=match_trans)

    # State 4: insert (emit output, equilibrium frequencies)
    insert_trans = []
    for j_idx, out in enumerate(AA_ALPHA):
        pj = pi_names[j_idx]
        # Self-loop
        insert_trans.append(MachineTransition(
            dest=4, weight={"*": ["r", pj]}, output=out))
        # Exit to begin
        insert_trans.append(MachineTransition(
            dest=0, weight=scaled(pj), output=out))
    insert = MachineState(name="insert", trans=insert_trans)

    # State 5: delete (consume input)
    delete_trans = []
    for inp in AA_ALPHA:
        # Self-loop
        delete_trans.append(MachineTransition(
            dest=5, weight="r", input=inp))
        # Exit to orphan (TKF91: delete -> orphan)
        delete_trans.append(MachineTransition(
            dest=1, weight={"not": "r"}, input=inp))
    delete = MachineState(name="delete", trans=delete_trans)

    # State 6: end
    end = MachineState(name="end", trans=[])

    return Machine(
        state=[begin, orphan, wait, match, insert, delete, end],
        defs=defs,
    )


try:
    import flax.linen as nn

    class TKF92Heads(nn.Module):
        """Parameter heads: (L, d_model) -> per-position TKF92 parameters.

        Outputs (all shape (L,) before padding):
            t:       softplus(Dense(1)) + 0.01          positive time
            insRate: softplus(Dense(1)) * 0.01           insertion rate
            delRate: insRate + softplus(Dense(1)) * 0.01 deletion rate > insRate
            r:       sigmoid(Dense(1))                   fragment extension
            pi:      softmax(Dense(20))                  equilibrium freqs
        """
        n_aa: int = N_AA

        @nn.compact
        def __call__(self, x):
            # x: (L, d_model)
            raw_t = nn.Dense(1)(x).squeeze(-1)
            t = jax.nn.softplus(raw_t) + 0.01

            raw_ins = nn.Dense(1)(x).squeeze(-1)
            insRate = jax.nn.softplus(raw_ins) * 0.01

            raw_del_extra = nn.Dense(1)(x).squeeze(-1)
            delRate = insRate + jax.nn.softplus(raw_del_extra) * 0.01

            raw_r = nn.Dense(1)(x).squeeze(-1)
            r = jax.nn.sigmoid(raw_r)

            raw_pi = nn.Dense(self.n_aa)(x)  # (L, 20)
            pi = jax.nn.softmax(raw_pi, axis=-1)

            return t, insRate, delRate, r, pi

    def heads_to_dp_params(t, insRate, delRate, r, pi):
        """Reshape parameter head outputs to (Li+1, 1) tensors for DP.

        Boundary row (index 0) uses the first position's values.
        Pi is split into 20 separate parameter tensors pi_0..pi_19.
        """
        def pad(x):
            return jnp.concatenate([x[:1], x])[:, None]

        params = {
            "t": pad(t),
            "insRate": pad(insRate),
            "delRate": pad(delRate),
            "r": pad(r),
        }
        # Split pi: (L, 20) -> 20 × (L,) -> pad each
        pi_padded = jnp.concatenate([pi[:1], pi], axis=0)  # (Li+1, 20)
        for j in range(pi.shape[-1]):
            params[f"pi_{j}"] = pi_padded[:, j:j+1]  # (Li+1, 1)

        return params

    def make_loss_fn(pm):
        """Create a loss function for TKF92: transformer params -> -log-likelihood.

        Args:
            pm: ParameterizedMachine from make_tkf92_machine().
        """
        from ..jax.dp_neural import neural_log_forward_tok

        def loss_fn(model_params, transformer, heads, msa_onehot,
                    input_tokens, output_tokens):
            # Run MSA transformer
            embeddings = transformer.apply(
                model_params["transformer"], msa_onehot)  # (L, d_model)
            # Run parameter heads
            t, insRate, delRate, r, pi = heads.apply(
                model_params["heads"], embeddings)
            dp_params = heads_to_dp_params(t, insRate, delRate, r, pi)
            ll = neural_log_forward_tok(pm, input_tokens, output_tokens, dp_params)
            return -ll

        return loss_fn

except ImportError:
    pass  # Flax not available; machine construction still works
