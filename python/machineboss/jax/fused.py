"""Fused Plan7+transducer DP algorithms.

Avoids explicit composition of the Plan7 profile HMM with a transducer
by interleaving their DP in a single pass. The Plan7 HMM generates
intermediate symbols (e.g. amino acids) which become inputs to the
transducer, which produces the final output (e.g. DNA).

Like GeneWise, this fused approach avoids materializing the huge
composite state space.

JIT-compilable: all loops use jax.lax.scan / jax.lax.while_loop.
Semiring-parameterized: same code for Forward (LOGSUMEXP) and Viterbi (MAXPLUS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ..eval import EvaluatedMachine, EvaluatedTransition
from .types import NEG_INF
from .semiring import LogSemiring, LOGSUMEXP, MAXPLUS


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


def _fused_dp(fused: FusedMachine, output_seq: jnp.ndarray,
              semiring: LogSemiring) -> float:
    """Fused Plan7+transducer DP (Forward or Viterbi via semiring).

    Uses jax.lax.scan over the output sequence. All inner loops are
    vectorized with JAX operations. JIT-compilable.

    Args:
        fused: FusedMachine
        output_seq: (Lo,) output token indices
        semiring: LOGSUMEXP for Forward, MAXPLUS for Viterbi
    Returns:
        Log-likelihood or Viterbi score (scalar).
    """
    S_p7 = fused.p7_n_states
    S_td = fused.td_n_states
    Lo = len(output_seq)
    n_p7_out = fused.p7_log_trans.shape[0]

    p7_log_trans = jnp.array(fused.p7_log_trans)
    p7_silent = jnp.array(fused.p7_silent)
    td_log_trans = jnp.array(fused.td_log_trans)
    p7_out_to_td_in = jnp.array(fused.p7_out_to_td_in)

    # Token mask: which Plan7 output tokens are emitting (> 0) and map to valid TD input
    emit_mask = (jnp.arange(n_p7_out) > 0) & (p7_out_to_td_in > 0)  # (n_p7_out,)

    def propagate_silent_composite(cell):
        """Propagate silent transitions in both Plan7 and transducer."""
        def body_fn(carry):
            prev, _ = carry

            # Plan7 silent: sum over p7_src
            p7_update = semiring.reduce(
                prev[:, None, :] + p7_silent[:, :, None], axis=0)  # (S_p7, S_td)

            # Transducer silent: sum over td_src
            td_silent = td_log_trans[0, 0]  # (S_td, S_td)
            td_update = semiring.reduce(
                prev[:, :, None] + td_silent[None, :, :], axis=1)  # (S_p7, S_td)

            new = semiring.plus(cell, semiring.plus(p7_update, td_update))
            return new, prev

        def cond_fn(carry):
            new, prev = carry
            return jnp.any(jnp.abs(new - prev) > 1e-10)

        init = (cell, jnp.full_like(cell, NEG_INF))
        result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
        return result

    def emit_produce(prev, out_tok):
        """Plan7 emits intermediate symbol, transducer produces output.

        Vectorized over all Plan7 output tokens.
        """
        # Gather transducer transitions for each P7 output token
        td_trans_emit = td_log_trans[p7_out_to_td_in, out_tok]  # (n_p7_out, S_td, S_td)

        # Plan7 propagation: prev[p7_src, td_src] + p7_trans[tok, p7_src, p7_dst]
        # → intermediate[tok, p7_dst, td_src]
        intermediate = semiring.reduce(
            prev[None, :, None, :] + p7_log_trans[:, :, :, None],
            axis=1)  # (n_p7_out, S_p7, S_td)

        # Transducer propagation: intermediate[tok, p7_dst, td_src] + td_trans[tok, td_src, td_dst]
        # → result[tok, p7_dst, td_dst]
        result = semiring.reduce(
            intermediate[:, :, :, None] + td_trans_emit[:, None, :, :],
            axis=2)  # (n_p7_out, S_p7, S_td)

        # Mask non-emitting tokens
        result = jnp.where(emit_mask[:, None, None], result, NEG_INF)

        # Reduce over tokens
        return semiring.reduce(result, axis=0)  # (S_p7, S_td)

    def td_delete(prev, out_tok):
        """Transducer produces output without Plan7 emission."""
        td_trans = td_log_trans[0, out_tok]  # (S_td, S_td)
        return semiring.reduce(
            prev[:, :, None] + td_trans[None, :, :], axis=1)  # (S_p7, S_td)

    def p7_emit_no_output(cell):
        """Plan7 emits, transducer consumes silently (no output produced)."""
        # Gather transducer silent-input transitions for each P7 output token
        td_trans_silent = td_log_trans[p7_out_to_td_in, 0]  # (n_p7_out, S_td, S_td)

        intermediate = semiring.reduce(
            cell[None, :, None, :] + p7_log_trans[:, :, :, None],
            axis=1)  # (n_p7_out, S_p7, S_td)

        result = semiring.reduce(
            intermediate[:, :, :, None] + td_trans_silent[:, None, :, :],
            axis=2)  # (n_p7_out, S_p7, S_td)

        result = jnp.where(emit_mask[:, None, None], result, NEG_INF)
        update = semiring.reduce(result, axis=0)  # (S_p7, S_td)
        return semiring.plus(cell, update)

    # Initialize: start state (0, 0)
    init_cell = jnp.full((S_p7, S_td), NEG_INF)
    init_cell = init_cell.at[0, 0].set(0.0)

    # Propagate silent → P7 emit (no output) → silent at position 0
    init_cell = propagate_silent_composite(init_cell)
    init_cell = p7_emit_no_output(init_cell)
    init_cell = propagate_silent_composite(init_cell)

    if Lo == 0:
        return init_cell[S_p7 - 1, S_td - 1]

    # Scan over output positions
    def scan_fn(prev, out_tok):
        # Transitions that produce output
        cell = emit_produce(prev, out_tok)
        cell = semiring.plus(cell, td_delete(prev, out_tok))

        # Silent and internal P7 emissions
        cell = propagate_silent_composite(cell)
        cell = p7_emit_no_output(cell)
        cell = propagate_silent_composite(cell)

        return cell, None

    final_cell, _ = jax.lax.scan(scan_fn, init_cell, output_seq)
    return final_cell[S_p7 - 1, S_td - 1]


def fused_log_forward(fused: FusedMachine,
                      output_seq: jnp.ndarray) -> float:
    """Fused Plan7+transducer Forward algorithm.

    Args:
        fused: FusedMachine
        output_seq: (Lo,) output token indices (in transducer's output alphabet)
    Returns:
        Log-likelihood (scalar).
    """
    return _fused_dp(fused, output_seq, LOGSUMEXP)


def fused_log_viterbi(fused: FusedMachine,
                      output_seq: jnp.ndarray) -> float:
    """Fused Plan7+transducer Viterbi algorithm.

    Args:
        fused: FusedMachine
        output_seq: (Lo,) output token indices (in transducer's output alphabet)
    Returns:
        Log-probability of most likely path (scalar).
    """
    return _fused_dp(fused, output_seq, MAXPLUS)
