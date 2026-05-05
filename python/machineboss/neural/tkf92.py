"""Neural TKF92 protein transducer: MSA transformer -> per-position WFST parameters.

The TKF92 conditional WFST is the 5-state machine derived in
tkf-mixdom/tkf/tkf92-wfst-derivation.tex by dividing the TKF92 Pair HMM
by the TKF92 singlet HMM. States: begin (= sta), match, insert, delete,
end (= fin). Each non-end state has direct transitions to mat/ins/del/fin
columns with WFST entries t_{a,b}, eliminating the silent intermediates
(wait, orphan) of the older Machine Boss form.

Free parameters:
    t                   evolutionary time
    insRate, delRate    BDI rates (require delRate > insRate)
    r                   fragment extension probability (= ext)
    pi_0..pi_19         amino acid equilibrium frequencies (F81 substitution)

Derived: alpha = pNoDeletion, beta = pDescendants, gamma = pOrphans,
kappa = insRate/delRate (singlet equilibrium continuation), p_singlet =
r + (1-r) * kappa (TKF92 singlet continuation). The WFST conditional
P(descendant | ancestor, theta) is recovered by composing tkf91root
(geometric singlet with parameter kappa) with this branch transducer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..machine import Machine, MachineState, MachineTransition

# 20 standard amino acids (alphabetical)
AA_ALPHA = sorted("ACDEFGHIKLMNPQRSTVWY")
N_AA = 20


def make_tkf92_machine() -> Machine:
    """Build the canonical 5-state TKF92 protein WFST with F81 substitution.

    States (index, id):
        0 begin   = sta  (before any column)
        1 match   = mat  (just emitted an ancestor->descendant match column)
        2 insert  = ins  (just emitted a descendant-only insertion column)
        3 delete  = del  (just consumed an ancestor-only deletion column)
        4 end     = fin

    For each non-end source a in {begin, match, insert, delete}, transitions
    are: a -> match (consume X, emit Y), a -> insert (emit Y), a -> delete
    (consume X), a -> end. Per-transition weight is t_{a,b} times the column
    emission weight.
    """
    # Equilibrium-frequency parameters named by amino-acid letter (pi_A, pi_C, ...).
    pi_names = [f"pi_{aa}" for aa in AA_ALPHA]

    # Standard TKF91 quantities + TKF92 singlet additions.
    defs = {
        "pNoDeletion":   {"exp": {"*": [-1, {"*": ["delRate", "t"]}]}},
        "pDeletion":     {"not": "pNoDeletion"},
        "pNoInsertion":  {"exp": {"*": [-1, {"*": ["insRate", "t"]}]}},
        "pInsertion":    {"not": "pNoInsertion"},
        "delInsRatio":   {"/": ["pNoDeletion", "pNoInsertion"]},
        "pDescendants":  {"/": [
            {"*": ["insRate", {"not": "delInsRatio"}]},
            {"-": ["delRate", {"*": ["insRate", "delInsRatio"]}]},
        ]},
        "pNoDescendants": {"not": "pDescendants"},
        "pOrphans":       {"not": {"*": [
            {"/": ["delRate", "insRate"]},
            {"/": ["pDescendants", "pDeletion"]},
        ]}},
        "pNoOrphans":     {"not": "pOrphans"},
        "pNoSub":         {"exp": {"*": [-1, "t"]}},
        "pSub":           {"not": "pNoSub"},

        # TKF92 singlet continuation
        "kappa":     {"/": ["insRate", "delRate"]},
        "pSinglet":  {"+": ["r", {"*": [{"not": "r"}, "kappa"]}]},

        # WFST transition entries t_{a,b} from the derivation.
        # sta row (no fragment-extension factor since fragment cannot extend before any column).
        "tStaMat": {"*": ["pNoDescendants", "pNoDeletion"]},          # (1-beta)*alpha
        "tStaIns": "pDescendants",                                     # beta
        "tStaDel": {"*": ["pNoDescendants", "pDeletion"]},             # (1-beta)*(1-alpha)
        "tStaFin": "pNoDescendants",                                   # 1-beta

        # mat row
        "tMatMat": {"/": [
            {"+": ["r", {"*": [{"not": "r"}, {"*": ["pNoDescendants", {"*": ["kappa", "pNoDeletion"]}]}]}]},
            "pSinglet",
        ]},
        "tMatIns": {"*": [{"not": "r"}, "pDescendants"]},               # (1-r)*beta
        "tMatDel": {"/": [
            {"*": [{"not": "r"}, {"*": ["pNoDescendants", {"*": ["kappa", "pDeletion"]}]}]},
            "pSinglet",
        ]},
        "tMatFin": "pNoDescendants",                                    # 1-beta

        # ins row
        "tInsMat": {"/": [
            {"*": [{"not": "r"}, {"*": ["pNoDescendants", {"*": ["kappa", "pNoDeletion"]}]}]},
            "pSinglet",
        ]},
        "tInsIns": {"+": ["r", {"*": [{"not": "r"}, "pDescendants"]}]}, # r + (1-r)*beta
        "tInsDel": "tMatDel",                                            # same as tMatDel
        "tInsFin": "pNoDescendants",

        # del row (uses gamma = pOrphans instead of beta)
        "tDelMat": {"/": [
            {"*": [{"not": "r"}, {"*": ["pNoOrphans", {"*": ["kappa", "pNoDeletion"]}]}]},
            "pSinglet",
        ]},
        "tDelIns": {"*": [{"not": "r"}, "pOrphans"]},                   # (1-r)*gamma
        "tDelDel": {"/": [
            {"+": ["r", {"*": [{"not": "r"}, {"*": ["pNoOrphans", {"*": ["kappa", "pDeletion"]}]}]}]},
            "pSinglet",
        ]},
        "tDelFin": "pNoOrphans",                                        # 1-gamma
    }

    # F81 emission weight for a match transition X -> Y.
    def emit_match(X, Y_idx):
        pj = pi_names[Y_idx]
        if X == AA_ALPHA[Y_idx]:
            return {"+": ["pNoSub", {"*": ["pSub", pj]}]}
        return {"*": ["pSub", pj]}

    # Build outgoing transitions for a source state with row weights
    # (tMat, tIns, tDel, tFin).
    def row_trans(t_mat, t_ins, t_del, t_fin):
        trans = []
        # source -> match (X | Y)
        for X in AA_ALPHA:
            for Y_idx, Y in enumerate(AA_ALPHA):
                w = {"*": [t_mat, emit_match(X, Y_idx)]}
                trans.append(MachineTransition(dest=1, input=X, output=Y, weight=w))
        # source -> insert (eps | Y)
        for Y_idx, Y in enumerate(AA_ALPHA):
            w = {"*": [t_ins, pi_names[Y_idx]]}
            trans.append(MachineTransition(dest=2, output=Y, weight=w))
        # source -> delete (X | eps)
        for X in AA_ALPHA:
            trans.append(MachineTransition(dest=3, input=X, weight=t_del))
        # source -> end (eps | eps)
        trans.append(MachineTransition(dest=4, weight=t_fin))
        return trans

    begin  = MachineState(name="begin",  trans=row_trans("tStaMat", "tStaIns", "tStaDel", "tStaFin"))
    match  = MachineState(name="match",  trans=row_trans("tMatMat", "tMatIns", "tMatDel", "tMatFin"))
    insert = MachineState(name="insert", trans=row_trans("tInsMat", "tInsIns", "tInsDel", "tInsFin"))
    delete = MachineState(name="delete", trans=row_trans("tDelMat", "tDelIns", "tDelDel", "tDelFin"))
    end    = MachineState(name="end",    trans=[])

    return Machine(state=[begin, match, insert, delete, end], defs=defs)


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
        # Split pi: (L, 20) -> 20 × (L,) -> pad each. Names are pi_A, pi_C, ...
        pi_padded = jnp.concatenate([pi[:1], pi], axis=0)  # (Li+1, 20)
        for j, aa in enumerate(AA_ALPHA):
            params[f"pi_{aa}"] = pi_padded[:, j:j+1]  # (Li+1, 1)

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
