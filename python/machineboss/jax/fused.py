"""Fused Plan7+transducer DP algorithms.

Avoids explicit composition of the Plan7 profile HMM with a transducer
by interleaving their DP in a single pass. The Plan7 HMM generates
intermediate symbols (e.g. amino acids) which become inputs to the
transducer, which produces the final output (e.g. DNA).

Like GeneWise, this fused approach avoids materializing the huge
composite state space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ..eval import EvaluatedMachine, EvaluatedTransition
from .types import NEG_INF


@dataclass
class FusedMachine:
    """Fused Plan7 + transducer representation for DP.

    The composite state is (plan7_state, transducer_state).
    Only reachable pairs are tracked.
    """
    # Plan7 model arrays
    p7_n_states: int
    p7_log_trans: np.ndarray   # (O_p7, S_p7, S_p7) log-weights indexed by [out_tok, src, dst]
    p7_silent: np.ndarray      # (S_p7, S_p7) silent transitions

    # Transducer arrays
    td_n_states: int
    td_log_trans: np.ndarray   # (I_td, O_td, S_td, S_td) log-weights

    # Alphabet mapping: Plan7 output alphabet = transducer input alphabet
    # p7_out_to_td_in[p7_out_tok] = td_in_tok
    p7_out_to_td_in: np.ndarray  # mapping from plan7 output tokens to transducer input tokens

    @classmethod
    def build(cls, plan7_em: EvaluatedMachine, transducer_em: EvaluatedMachine) -> FusedMachine:
        """Build fused representation from evaluated Plan7 and transducer machines.

        Plan7 is a generator (no input, output = intermediate alphabet).
        Transducer has input = intermediate alphabet, output = final alphabet.
        """
        S_p7 = plan7_em.n_states
        S_td = transducer_em.n_states

        # Build Plan7 transition tensor: (n_out_tokens, S_p7, S_p7)
        n_out_p7 = len(plan7_em.output_tokens)
        p7_log_trans = np.full((n_out_p7, S_p7, S_p7), NEG_INF, dtype=np.float32)
        for t in plan7_em.transitions:
            cur = p7_log_trans[t.out_tok, t.src, t.dst]
            if cur == NEG_INF:
                p7_log_trans[t.out_tok, t.src, t.dst] = t.log_weight
            else:
                p7_log_trans[t.out_tok, t.src, t.dst] = np.logaddexp(cur, t.log_weight)

        p7_silent = p7_log_trans[0]  # silent transitions (out_tok=0)

        # Build transducer transition tensor: (n_in_td, n_out_td, S_td, S_td)
        n_in_td = len(transducer_em.input_tokens)
        n_out_td = len(transducer_em.output_tokens)
        td_log_trans = np.full((n_in_td, n_out_td, S_td, S_td), NEG_INF, dtype=np.float32)
        for t in transducer_em.transitions:
            cur = td_log_trans[t.in_tok, t.out_tok, t.src, t.dst]
            if cur == NEG_INF:
                td_log_trans[t.in_tok, t.out_tok, t.src, t.dst] = t.log_weight
            else:
                td_log_trans[t.in_tok, t.out_tok, t.src, t.dst] = np.logaddexp(cur, t.log_weight)

        # Map Plan7 output tokens to transducer input tokens
        td_in_map = {sym: i for i, sym in enumerate(transducer_em.input_tokens)}
        p7_out_to_td_in = np.zeros(n_out_p7, dtype=np.int32)
        for p7_tok, sym in enumerate(plan7_em.output_tokens):
            if sym in td_in_map:
                p7_out_to_td_in[p7_tok] = td_in_map[sym]
            # else: 0 (empty token) — silent in transducer input

        return cls(
            p7_n_states=S_p7,
            p7_log_trans=p7_log_trans,
            p7_silent=p7_silent,
            td_n_states=S_td,
            td_log_trans=td_log_trans,
            p7_out_to_td_in=p7_out_to_td_in,
        )


def fused_log_forward(fused: FusedMachine,
                      output_seq: jnp.ndarray) -> float:
    """Fused Plan7+transducer Forward algorithm.

    The composite machine is a generator (no explicit input).
    Plan7 generates intermediate symbols, transducer converts to output.

    Args:
        fused: FusedMachine
        output_seq: (Lo,) output token indices (in transducer's output alphabet)

    Returns:
        Log-likelihood (scalar).
    """
    S_p7 = fused.p7_n_states
    S_td = fused.td_n_states
    Lo = len(output_seq)

    p7_log_trans = jnp.array(fused.p7_log_trans)
    p7_silent = jnp.array(fused.p7_silent)
    td_log_trans = jnp.array(fused.td_log_trans)
    p7_out_to_td_in = jnp.array(fused.p7_out_to_td_in)

    # DP matrix: dp[outPos, p7_state, td_state] in log-space
    dp = jnp.full((Lo + 1, S_p7, S_td), NEG_INF)

    # Start: Plan7 state 0, transducer state 0, output position 0
    dp = dp.at[0, 0, 0].set(0.0)

    def propagate_silent_composite(cell):
        """Propagate silent transitions in both Plan7 and transducer."""
        # Iterate until convergence
        def body_fn(carry):
            prev, _ = carry

            # Plan7 silent transitions (no emission)
            # For each (p7_src, td_s): sum over p7_dst of prev[p7_src, td_s] + p7_silent[p7_src, p7_dst]
            p7_incoming = prev[:, None, :] + p7_silent[:, :, None]  # (S_p7, S_p7, S_td)
            p7_update = jax.nn.logsumexp(p7_incoming, axis=0)  # (S_p7, S_td)

            # Transducer silent transitions (no input, no output)
            td_silent = td_log_trans[0, 0]  # (S_td, S_td)
            td_incoming = prev[:, :, None] + td_silent[None, :, :]  # (S_p7, S_td, S_td)
            td_update = jax.nn.logsumexp(td_incoming, axis=1)  # (S_p7, S_td)

            new = jnp.logaddexp(cell, jnp.logaddexp(p7_update, td_update))
            return new, prev

        def cond_fn(carry):
            new, prev = carry
            return jnp.any(jnp.abs(new - prev) > 1e-10)

        init = (cell, jnp.full_like(cell, NEG_INF))
        result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
        return result

    # Propagate silent at position 0
    dp = dp.at[0].set(propagate_silent_composite(dp[0]))

    # Also propagate Plan7 emissions that feed into transducer
    # (Plan7 emits intermediate symbol, transducer consumes it silently or with output)
    def propagate_p7_emission(cell, prev_cell):
        """Plan7 emits intermediate symbol, transducer consumes it.

        This handles the case where Plan7 emits (advancing the Plan7 state)
        and the transducer consumes that emission as input (advancing the transducer state).
        No output is produced yet.
        """
        update = jnp.full_like(cell, NEG_INF)

        # For each Plan7 output token (intermediate symbol):
        n_p7_out = fused.p7_log_trans.shape[0]
        for p7_out_tok in range(1, n_p7_out):  # skip 0 (silent)
            td_in_tok = int(fused.p7_out_to_td_in[p7_out_tok])
            if td_in_tok == 0:
                continue

            # Plan7 transition with this emission: p7_log_trans[p7_out_tok, p7_src, p7_dst]
            p7_trans = p7_log_trans[p7_out_tok]  # (S_p7, S_p7)

            # Transducer transition consuming this input without output:
            td_trans = td_log_trans[td_in_tok, 0]  # (S_td, S_td): input=td_in_tok, output=0

            # Combine: for each (p7_src, td_src) -> (p7_dst, td_dst)
            # log_weight = prev[p7_src, td_src] + p7_trans[p7_src, p7_dst] + td_trans[td_src, td_dst]
            # Sum over p7_src and td_src
            # Result shape: (S_p7_dst, S_td_dst)
            combined = (prev_cell[:, None, :, None]  # (S_p7, 1, S_td, 1)
                        + p7_trans[:, :, None, None]  # (S_p7, S_p7, 1, 1)
                        + td_trans[None, None, :, :])  # (1, 1, S_td, S_td)
            # Sum over p7_src (axis 0) and td_src (axis 2)
            result = jax.nn.logsumexp(combined, axis=(0, 2))  # (S_p7, S_td)
            update = jnp.logaddexp(update, result)

        return jnp.logaddexp(cell, update)

    # Fill DP matrix
    for o in range(Lo + 1):
        if o == 0:
            # Also propagate P7 emissions at position 0 (they don't produce output)
            dp = dp.at[0].set(propagate_p7_emission(dp[0], dp[0]))
            dp = dp.at[0].set(propagate_silent_composite(dp[0]))
            continue

        cell = dp[o]
        out_tok = output_seq[o - 1]

        # Transitions that produce output[o-1]:
        # Case 1: Plan7 emits intermediate, transducer produces output
        n_p7_out = fused.p7_log_trans.shape[0]
        for p7_out_tok in range(1, n_p7_out):
            td_in_tok = int(fused.p7_out_to_td_in[p7_out_tok])
            if td_in_tok == 0:
                continue

            p7_trans = p7_log_trans[p7_out_tok]
            td_trans = td_log_trans[td_in_tok, out_tok]

            combined = (dp[o - 1, :, None, :, None]
                        + p7_trans[:, :, None, None]
                        + td_trans[None, None, :, :])
            result = jax.nn.logsumexp(combined, axis=(0, 2))
            cell = jnp.logaddexp(cell, result)

        # Case 2: Transducer produces output without consuming Plan7 emission
        # (transducer delete: input=0, output=out_tok)
        td_trans_delete = td_log_trans[0, out_tok]  # (S_td, S_td)
        td_incoming = dp[o - 1, :, :, None] + td_trans_delete[None, :, :]
        td_update = jax.nn.logsumexp(td_incoming, axis=1)
        cell = jnp.logaddexp(cell, td_update)

        # Propagate silent and internal Plan7 emissions
        cell = propagate_silent_composite(cell)
        cell = propagate_p7_emission(cell, cell)
        cell = propagate_silent_composite(cell)
        dp = dp.at[o].set(cell)

    # Result: end states of both Plan7 and transducer
    return dp[Lo, S_p7 - 1, S_td - 1]


def fused_log_viterbi(fused: FusedMachine,
                      output_seq: jnp.ndarray) -> float:
    """Fused Plan7+transducer Viterbi algorithm.

    Same structure as fused Forward but uses max instead of logsumexp.

    Args:
        fused: FusedMachine
        output_seq: (Lo,) output token indices (in transducer's output alphabet)

    Returns:
        Log-probability of most likely path (scalar).
    """
    S_p7 = fused.p7_n_states
    S_td = fused.td_n_states
    Lo = len(output_seq)

    p7_log_trans = jnp.array(fused.p7_log_trans)
    p7_silent = jnp.array(fused.p7_silent)
    td_log_trans = jnp.array(fused.td_log_trans)

    dp = jnp.full((Lo + 1, S_p7, S_td), NEG_INF)
    dp = dp.at[0, 0, 0].set(0.0)

    def propagate_silent_max(cell):
        """Propagate silent transitions using max (Viterbi)."""
        def body_fn(carry):
            prev, _ = carry
            p7_incoming = prev[:, None, :] + p7_silent[:, :, None]
            p7_update = jnp.max(p7_incoming, axis=0)

            td_silent = td_log_trans[0, 0]
            td_incoming = prev[:, :, None] + td_silent[None, :, :]
            td_update = jnp.max(td_incoming, axis=1)

            new = jnp.maximum(cell, jnp.maximum(p7_update, td_update))
            return new, prev

        def cond_fn(carry):
            new, prev = carry
            return jnp.any(jnp.abs(new - prev) > 1e-10)

        init = (cell, jnp.full_like(cell, NEG_INF))
        result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
        return result

    dp = dp.at[0].set(propagate_silent_max(dp[0]))

    def propagate_p7_emission_max(cell, prev_cell):
        """Plan7 emits, transducer consumes without output — max version."""
        update = jnp.full_like(cell, NEG_INF)
        n_p7_out = fused.p7_log_trans.shape[0]
        for p7_out_tok in range(1, n_p7_out):
            td_in_tok = int(fused.p7_out_to_td_in[p7_out_tok])
            if td_in_tok == 0:
                continue
            p7_trans = p7_log_trans[p7_out_tok]
            td_trans = td_log_trans[td_in_tok, 0]
            combined = (prev_cell[:, None, :, None]
                        + p7_trans[:, :, None, None]
                        + td_trans[None, None, :, :])
            result = jnp.max(combined, axis=(0, 2))
            update = jnp.maximum(update, result)
        return jnp.maximum(cell, update)

    for o in range(Lo + 1):
        if o == 0:
            dp = dp.at[0].set(propagate_p7_emission_max(dp[0], dp[0]))
            dp = dp.at[0].set(propagate_silent_max(dp[0]))
            continue

        cell = dp[o]
        out_tok = output_seq[o - 1]

        n_p7_out = fused.p7_log_trans.shape[0]
        for p7_out_tok in range(1, n_p7_out):
            td_in_tok = int(fused.p7_out_to_td_in[p7_out_tok])
            if td_in_tok == 0:
                continue
            p7_trans = p7_log_trans[p7_out_tok]
            td_trans = td_log_trans[td_in_tok, out_tok]
            combined = (dp[o - 1, :, None, :, None]
                        + p7_trans[:, :, None, None]
                        + td_trans[None, None, :, :])
            result = jnp.max(combined, axis=(0, 2))
            cell = jnp.maximum(cell, result)

        td_trans_delete = td_log_trans[0, out_tok]
        td_incoming = dp[o - 1, :, :, None] + td_trans_delete[None, :, :]
        td_update = jnp.max(td_incoming, axis=1)
        cell = jnp.maximum(cell, td_update)

        cell = propagate_silent_max(cell)
        cell = propagate_p7_emission_max(cell, cell)
        cell = propagate_silent_max(cell)
        dp = dp.at[o].set(cell)

    return dp[Lo, S_p7 - 1, S_td - 1]
