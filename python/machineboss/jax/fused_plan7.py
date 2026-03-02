"""Plan7-aware fused DP with nested scans.

Exploits the linear chain structure of Plan7 profile HMMs for efficient
fused DP with a transducer. Takes an HmmerModel directly (not a generic
Machine) to access the Plan7 topology.

Architecture:
  - Outer jax.lax.scan over output sequence positions
  - Inner jax.lax.scan over core profile nodes k=1..K
  - Flanking states (N, C, J) handled separately at each outer step

The composite state between output positions is:
  - core_m[k, td]: probability of being at M_k x transducer state td
  - core_d[k, td]: probability of being at D_k x transducer state td
  - flanking[f, td]: probability of being at flanking state f x td

The inner scan carries (m_entering, d_entering) x S_td through core
positions, which is O(S_td) per step instead of O(S_p7 * S_td).

Semiring-parameterized: same code for Forward (LOGSUMEXP) and Viterbi (MAXPLUS).
JIT-compilable: all loops use jax.lax.scan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..hmmer import HmmerModel
from ..eval import EvaluatedMachine
from .types import NEG_INF
from .semiring import LogSemiring, LOGSUMEXP, MAXPLUS


# Flanking state indices (within the flanking array)
_N = 0
_NX = 1
_B = 2
_E = 3
_CX = 4
_C = 5
_JX = 6
_J = 7
N_FLANKING = 8


@dataclass
class FusedPlan7Machine:
    """Plan7-aware fused representation for nested-scan DP.

    Stores the Plan7 profile structure explicitly so the inner scan
    can iterate over core nodes k=1..K.
    """
    K: int                      # number of profile nodes
    n_aa: int                   # number of amino acids (Plan7 output alphabet)
    S_td: int                   # number of transducer states
    n_out_td: int               # number of transducer output tokens

    # Per-node core transition log-weights, shape (K,)
    log_m_to_m: np.ndarray
    log_m_to_i: np.ndarray
    log_m_to_d: np.ndarray
    log_i_to_m: np.ndarray
    log_i_to_i: np.ndarray
    log_d_to_m: np.ndarray
    log_d_to_d: np.ndarray

    # Per-node emission log-weights
    log_match_emit: np.ndarray   # (K, n_aa)
    log_ins_emit: np.ndarray     # (K, n_aa) — I_k for k=1..K (I_0 not used)

    # Begin transitions (B → core), shape (K,) for local entry
    log_b_entry: np.ndarray      # B → M_k log-weight for each k

    # Flanking log-weights
    log_n_loop: float            # Nx → N
    log_n_to_b: float            # Nx → B
    log_e_to_cx: float           # E → Cx
    log_e_to_jx: float           # E → Jx (NEG_INF for single-hit)
    log_c_loop: float            # Cx → C
    log_c_to_t: float            # Cx → T
    log_j_loop: float            # Jx → J (NEG_INF for single-hit)
    log_j_to_b: float            # Jx → B (NEG_INF for single-hit)

    # Flanking emission log-weights (background), shape (n_aa,)
    log_null_emit: np.ndarray

    # Transducer: (n_in_td, n_out_td, S_td, S_td)
    td_log_trans: np.ndarray

    # Alphabet mapping: aa_index (1-based in Plan7 output) → td_input_token
    aa_to_td_in: np.ndarray      # (n_aa,)

    # Transducer silent transitions
    td_silent: np.ndarray        # (S_td, S_td) = td_log_trans[0, 0]

    @classmethod
    def build(cls, hmmer: HmmerModel, transducer_em: EvaluatedMachine,
              *, multihit: bool = False, L: float = 400) -> FusedPlan7Machine:
        """Build from HmmerModel and transducer EvaluatedMachine."""
        K = len(hmmer.nodes)
        n_aa = len(hmmer.alph)
        S_td = transducer_em.n_states

        # Core transitions (log-space)
        log_m_to_m = np.array([math.log(n.m_to_m) if n.m_to_m > 0 else NEG_INF
                               for n in hmmer.nodes], dtype=np.float32)
        log_m_to_i = np.array([math.log(n.m_to_i) if n.m_to_i > 0 else NEG_INF
                               for n in hmmer.nodes], dtype=np.float32)
        log_m_to_d = np.array([math.log(n.m_to_d) if n.m_to_d > 0 else NEG_INF
                               for n in hmmer.nodes], dtype=np.float32)
        log_i_to_m = np.array([math.log(n.i_to_m) if n.i_to_m > 0 else NEG_INF
                               for n in hmmer.nodes], dtype=np.float32)
        log_i_to_i = np.array([math.log(n.i_to_i) if n.i_to_i > 0 else NEG_INF
                               for n in hmmer.nodes], dtype=np.float32)
        log_d_to_m = np.array([math.log(n.d_to_m) if n.d_to_m > 0 else NEG_INF
                               for n in hmmer.nodes], dtype=np.float32)
        log_d_to_d = np.array([math.log(n.d_to_d) if n.d_to_d > 0 else NEG_INF
                               for n in hmmer.nodes], dtype=np.float32)

        # Emissions
        log_match_emit = np.array(
            [[math.log(e) if e > 0 else NEG_INF for e in n.match_emit]
             for n in hmmer.nodes], dtype=np.float32)  # (K, n_aa)
        log_ins_emit = np.array(
            [[math.log(e) if e > 0 else NEG_INF for e in n.ins_emit]
             for n in hmmer.nodes], dtype=np.float32)  # (K, n_aa)

        # Begin transitions (local mode: occupancy-weighted entry)
        occ = hmmer.calc_match_occupancy()
        Z = sum(occ[k] * (K - k + 1) for k in range(1, K))
        if Z > 0:
            log_b_entry = np.array(
                [math.log(occ[k] / Z) if occ[k] > 0 else NEG_INF
                 for k in range(1, K)] + [NEG_INF],  # last node: no entry in local
                dtype=np.float32)
        else:
            log_b_entry = np.full(K, NEG_INF, dtype=np.float32)

        # Flanking
        log_n_loop = math.log(L / (L + 1))
        log_n_to_b = math.log(1.0 / (L + 1))
        log_c_loop = math.log(L / (L + 1))
        log_c_to_t = math.log(1.0 / (L + 1))

        if multihit:
            log_e_to_cx = math.log(0.5)
            log_e_to_jx = math.log(0.5)
            log_j_loop = math.log(L / (L + 1))
            log_j_to_b = math.log(1.0 / (L + 1))
        else:
            log_e_to_cx = 0.0  # log(1.0)
            log_e_to_jx = NEG_INF
            log_j_loop = NEG_INF
            log_j_to_b = NEG_INF

        log_null_emit = np.array(
            [math.log(e) if e > 0 else NEG_INF for e in hmmer.null_emit],
            dtype=np.float32)

        # Build transducer tensor
        n_in_td = len(transducer_em.input_tokens)
        n_out_td = len(transducer_em.output_tokens)
        td_log_trans = np.full((n_in_td, n_out_td, S_td, S_td), NEG_INF, dtype=np.float32)
        for t in transducer_em.transitions:
            cur = td_log_trans[t.in_tok, t.out_tok, t.src, t.dst]
            if cur == NEG_INF:
                td_log_trans[t.in_tok, t.out_tok, t.src, t.dst] = t.log_weight
            else:
                td_log_trans[t.in_tok, t.out_tok, t.src, t.dst] = np.logaddexp(cur, t.log_weight)

        # Map amino acid index → transducer input token
        td_in_map = {sym: i for i, sym in enumerate(transducer_em.input_tokens)}
        aa_to_td_in = np.zeros(n_aa, dtype=np.int32)
        for aa_idx, sym in enumerate(hmmer.alph):
            if sym in td_in_map:
                aa_to_td_in[aa_idx] = td_in_map[sym]

        return cls(
            K=K, n_aa=n_aa, S_td=S_td, n_out_td=n_out_td,
            log_m_to_m=log_m_to_m, log_m_to_i=log_m_to_i, log_m_to_d=log_m_to_d,
            log_i_to_m=log_i_to_m, log_i_to_i=log_i_to_i,
            log_d_to_m=log_d_to_m, log_d_to_d=log_d_to_d,
            log_match_emit=log_match_emit, log_ins_emit=log_ins_emit,
            log_b_entry=log_b_entry,
            log_n_loop=log_n_loop, log_n_to_b=log_n_to_b,
            log_e_to_cx=log_e_to_cx, log_e_to_jx=log_e_to_jx,
            log_c_loop=log_c_loop, log_c_to_t=log_c_to_t,
            log_j_loop=log_j_loop, log_j_to_b=log_j_to_b,
            log_null_emit=log_null_emit,
            td_log_trans=td_log_trans,
            aa_to_td_in=aa_to_td_in,
            td_silent=td_log_trans[0, 0],
        )


def _td_matvec(td_trans, v, semiring):
    """Transducer matvec: result[dst] = reduce_src(v[src] + td_trans[src, dst])."""
    return semiring.reduce(v[:, None] + td_trans, axis=0)


def _td_emit_produce(v_td, aa_emit_log, aa_to_td_in, td_log_trans, out_tok, semiring):
    """Emit amino acid via Plan7, transducer consumes and produces output.

    Args:
        v_td: (S_td,) current transducer state values
        aa_emit_log: (n_aa,) log emission weights for this Plan7 state
        aa_to_td_in: (n_aa,) mapping to transducer input tokens
        td_log_trans: (n_in_td, n_out_td, S_td, S_td)
        out_tok: output token index
        semiring: LogSemiring
    Returns:
        (S_td,) updated transducer state values
    """
    # For each amino acid: emission_weight + td_trans[aa_td_in, out_tok] @ v_td
    # Then reduce over amino acids
    n_aa = aa_emit_log.shape[0]

    # Gather transducer transitions for each aa
    td_trans_per_aa = td_log_trans[aa_to_td_in, out_tok]  # (n_aa, S_td, S_td)

    # v_td[src] + td_trans[src, dst] → (n_aa, S_td)
    td_results = jax.vmap(
        lambda tt: _td_matvec(tt, v_td, semiring)
    )(td_trans_per_aa)  # (n_aa, S_td)

    # Weight by emission probability
    weighted = aa_emit_log[:, None] + td_results  # (n_aa, S_td)

    return semiring.reduce(weighted, axis=0)  # (S_td,)


def _td_emit_silent(v_td, aa_emit_log, aa_to_td_in, td_log_trans, semiring):
    """Emit amino acid via Plan7, transducer consumes but produces no output.

    Same as _td_emit_produce but with out_tok=0 (silent output).
    """
    td_trans_per_aa = td_log_trans[aa_to_td_in, 0]  # (n_aa, S_td, S_td)
    td_results = jax.vmap(
        lambda tt: _td_matvec(tt, v_td, semiring)
    )(td_trans_per_aa)  # (n_aa, S_td)
    weighted = aa_emit_log[:, None] + td_results  # (n_aa, S_td)
    return semiring.reduce(weighted, axis=0)  # (S_td,)


def _propagate_td_silent(v_td, td_silent, semiring):
    """Propagate transducer silent transitions to fixed point."""
    def body_fn(carry):
        prev, _ = carry
        update = _td_matvec(td_silent, prev, semiring)
        new = semiring.plus(v_td, update)
        return new, prev

    def cond_fn(carry):
        new, prev = carry
        return jnp.any(jnp.abs(new - prev) > 1e-10)

    init = (v_td, jnp.full_like(v_td, NEG_INF))
    result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return result


def _fused_plan7_dp(fm: FusedPlan7Machine, output_seq: jnp.ndarray,
                    semiring: LogSemiring) -> float:
    """Fused Plan7+transducer DP with nested scans.

    Outer scan: over output positions.
    Inner scan: over core profile nodes k=1..K.
    Flanking states handled separately at each outer step.
    """
    K = fm.K
    S_td = fm.S_td

    # Convert to JAX arrays
    log_m_to_m = jnp.array(fm.log_m_to_m)
    log_m_to_i = jnp.array(fm.log_m_to_i)
    log_m_to_d = jnp.array(fm.log_m_to_d)
    log_i_to_m = jnp.array(fm.log_i_to_m)
    log_i_to_i = jnp.array(fm.log_i_to_i)
    log_d_to_m = jnp.array(fm.log_d_to_m)
    log_d_to_d = jnp.array(fm.log_d_to_d)
    log_match_emit = jnp.array(fm.log_match_emit)
    log_ins_emit = jnp.array(fm.log_ins_emit)
    log_b_entry = jnp.array(fm.log_b_entry)
    log_null_emit = jnp.array(fm.log_null_emit)
    td_log_trans = jnp.array(fm.td_log_trans)
    aa_to_td_in = jnp.array(fm.aa_to_td_in)
    td_silent = jnp.array(fm.td_silent)

    Lo = len(output_seq)

    # State representation:
    #   core_m: (K, S_td) — at M_k post-emission (ready for Mx_k routing)
    #   core_i: (K, S_td) — at I_k post-emission (ready for Ix_k routing)
    #   core_d: (K, S_td) — at D_k (silent)
    #   flanking: (N_FLANKING, S_td)

    def _init_state():
        """Initialize: start at S, then S → Nx (weight 1.0)."""
        core_m = jnp.full((K, S_td), NEG_INF)
        core_i = jnp.full((K, S_td), NEG_INF)
        core_d = jnp.full((K, S_td), NEG_INF)
        flanking = jnp.full((N_FLANKING, S_td), NEG_INF)
        # S → Nx with weight 1.0; td starts at state 0
        flanking = flanking.at[_NX, 0].set(0.0)
        return core_m, core_i, core_d, flanking

    def _propagate_flanking_silent(flanking, semiring):
        """Propagate silent flanking transitions: Nx→N/B, Cx→C/T, Jx→J/B, E→Cx/Jx.

        Also propagates transducer silent transitions at each flanking state.
        """
        def body_fn(carry):
            fl, _ = carry

            new_fl = jnp.full_like(fl, NEG_INF)

            # Nx → B: silent
            nx_to_b = fl[_NX] + fm.log_n_to_b
            new_fl = new_fl.at[_B].set(
                semiring.plus(new_fl[_B], nx_to_b))

            # Cx → T: silent (T is terminal, not tracked)
            # E → Cx: silent
            e_to_cx = fl[_E] + fm.log_e_to_cx
            new_fl = new_fl.at[_CX].set(
                semiring.plus(new_fl[_CX], e_to_cx))

            # E → Jx: silent
            e_to_jx = fl[_E] + fm.log_e_to_jx
            new_fl = new_fl.at[_JX].set(
                semiring.plus(new_fl[_JX], e_to_jx))

            # Jx → B: silent
            jx_to_b = fl[_JX] + fm.log_j_to_b
            new_fl = new_fl.at[_B].set(
                semiring.plus(new_fl[_B], jx_to_b))

            # Propagate td silent at each flanking state
            new_fl = jax.vmap(
                lambda v: _propagate_td_silent(v, td_silent, semiring)
            )(new_fl)

            combined = semiring.plus(fl, new_fl)
            return combined, fl

        def cond_fn(carry):
            new, prev = carry
            return jnp.any(jnp.abs(new - prev) > 1e-10)

        init = (flanking, jnp.full_like(flanking, NEG_INF))
        result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
        return result

    def _enter_core_from_b(b_val, semiring):
        """B → M_k entries (local mode). Returns initial core_m, core_d."""
        # B → M_k with log_b_entry[k]
        # Each M_k starts with b_val + log_b_entry[k], at the M_k state
        # (which will then emit — but emission happens during the output step)
        #
        # Actually, B → M_k means we go to M_k which needs to emit.
        # Before any output, these M_k values are "pre-emission" states.
        # In the x-state pattern: B → m_idx(k), which is the emitting state.
        # The emission will happen when we process an output token.
        core_m = log_b_entry[:, None] + b_val[None, :]  # (K, S_td)
        core_d = jnp.full((K, S_td), NEG_INF)
        return core_m, core_d

    def _propagate_core_silent(core_m, core_d, semiring):
        """Propagate silent transitions within core: D chains and routing.

        After emission, M_k goes to Mx_k which routes to M_{k+1}, I_k, D_{k+1}.
        D_k routes to M_{k+1}, D_{k+1}.
        This is the "post-emission routing" + delete chain propagation.

        Uses inner scan over k=1..K to propagate the delete chain.
        Returns updated core_m, core_d, and E accumulator.
        """
        # The post-emission routing from Mx_k:
        #   Mx_k → M_{k+1}: weight m_to_m[k]  (goes to core_m[k+1] pre-emission)
        #   Mx_k → I_k: weight m_to_i[k]  (goes to core_i[k] pre-emission)
        #   Mx_k → D_{k+1}: weight m_to_d[k]  (goes to core_d[k+1])
        #
        # Similarly Ix_k routing:
        #   Ix_k → M_{k+1}: weight i_to_m[k]
        #   Ix_k → I_k: weight i_to_i[k]  (self-loop: handled during emission)
        #
        # D_k routing:
        #   D_k → M_{k+1}: weight d_to_m[k]
        #   D_k → D_{k+1}: weight d_to_d[k]
        #
        # In local mode, M_k and D_k also → E (weight 1.0 additive)

        # First, compute what Mx_k and D_k contribute to the next position
        # and to E. We scan forward through k.

        # For Mx_k: source is core_m[k] (post-emission)
        # For D_k: source is core_d[k]

        # Contributions to E (local mode: all M_k and D_k can exit)
        e_from_m = core_m  # M_k → E with weight 1 (log 0)
        e_from_d = core_d  # D_k → E with weight 1 (log 0)

        # Contributions to next core positions via inner scan
        def inner_scan_fn(carry, k):
            """Process node k: propagate incoming m/d to next position."""
            m_incoming, d_incoming = carry  # (S_td,) each

            # m_incoming is what flows into M_k from the left
            # d_incoming is what flows into D_k from the left
            # Plus whatever is already in core_m[k] and core_d[k]
            m_at_k = semiring.plus(core_m[k], m_incoming)
            d_at_k = semiring.plus(core_d[k], d_incoming)

            # Propagate td silent at M_k and D_k
            m_at_k = _propagate_td_silent(m_at_k, td_silent, semiring)
            d_at_k = _propagate_td_silent(d_at_k, td_silent, semiring)

            # Mx_k routing (from m_at_k):
            m_to_next = m_at_k + log_m_to_m[k]  # → M_{k+1}
            d_to_next_from_m = m_at_k + log_m_to_d[k]  # → D_{k+1}

            # D_k routing:
            m_to_next_from_d = d_at_k + log_d_to_m[k]  # → M_{k+1}
            d_to_next_from_d = d_at_k + log_d_to_d[k]  # → D_{k+1}

            # Combined outgoing
            m_out = semiring.plus(m_to_next, m_to_next_from_d)
            d_out = semiring.plus(d_to_next_from_m, d_to_next_from_d)

            return (m_out, d_out), (m_at_k, d_at_k)

        # For k=0 (first node), incoming is from B entry (already in core_m)
        init_carry = (jnp.full(S_td, NEG_INF), jnp.full(S_td, NEG_INF))
        (m_exit, d_exit), (all_m, all_d) = jax.lax.scan(
            inner_scan_fn, init_carry, jnp.arange(K))

        # E accumulator: in local mode, every M_k and D_k can go to E
        e_val = semiring.reduce(
            jnp.stack([
                semiring.reduce(all_m, axis=0),
                semiring.reduce(all_d, axis=0),
            ], axis=0), axis=0)  # (S_td,)

        # m_exit and d_exit go to E in non-local mode
        # (In local mode they already contributed via all_m/all_d)

        return all_m, all_d, e_val

    def _emit_output_step(core_m, core_i, core_d, flanking, out_tok, semiring):
        """Process one output token: Plan7 emits → transducer produces output.

        Returns updated state after emission + silent propagation.
        """
        new_core_m = jnp.full((K, S_td), NEG_INF)
        new_core_i = jnp.full((K, S_td), NEG_INF)
        new_core_d = jnp.full((K, S_td), NEG_INF)
        new_flanking = jnp.full((N_FLANKING, S_td), NEG_INF)

        # 1. Core M_k emits amino acid, transducer produces output
        def m_emit(k_data):
            m_val, m_emit_log = k_data
            return _td_emit_produce(
                m_val, m_emit_log, aa_to_td_in, td_log_trans, out_tok, semiring)

        m_emitted = jax.vmap(m_emit)((core_m, log_match_emit))  # (K, S_td)

        # 2. Core I_k emits amino acid, transducer produces output
        def i_emit(k_data):
            i_val, i_emit_log = k_data
            return _td_emit_produce(
                i_val, i_emit_log, aa_to_td_in, td_log_trans, out_tok, semiring)

        i_emitted = jax.vmap(i_emit)((core_i, log_ins_emit))  # (K, S_td)

        # 3. Flanking N/C/J emit background, transducer produces output
        n_emitted = _td_emit_produce(
            flanking[_N], log_null_emit, aa_to_td_in, td_log_trans, out_tok, semiring)
        c_emitted = _td_emit_produce(
            flanking[_C], log_null_emit, aa_to_td_in, td_log_trans, out_tok, semiring)
        j_emitted = _td_emit_produce(
            flanking[_J], log_null_emit, aa_to_td_in, td_log_trans, out_tok, semiring)

        # 4. Transducer produces output without Plan7 emission (td silent input)
        td_delete = td_log_trans[0, out_tok]  # (S_td, S_td)

        def apply_td_delete(v):
            return _td_matvec(td_delete, v, semiring)

        # Apply td_delete to all states
        td_del_core_m = jax.vmap(apply_td_delete)(core_m)  # (K, S_td)
        td_del_core_i = jax.vmap(apply_td_delete)(core_i)
        td_del_core_d = jax.vmap(apply_td_delete)(core_d)
        td_del_flanking = jax.vmap(apply_td_delete)(flanking)

        # Combine: emitted + td_delete contributions
        new_core_m = semiring.plus(m_emitted, td_del_core_m)
        new_core_i = semiring.plus(i_emitted, td_del_core_i)
        new_core_d = td_del_core_d  # D states don't emit in Plan7
        new_flanking = new_flanking.at[_N].set(
            semiring.plus(n_emitted, td_del_flanking[_N]))
        new_flanking = new_flanking.at[_NX].set(td_del_flanking[_NX])
        new_flanking = new_flanking.at[_B].set(td_del_flanking[_B])
        new_flanking = new_flanking.at[_E].set(td_del_flanking[_E])
        new_flanking = new_flanking.at[_CX].set(td_del_flanking[_CX])
        new_flanking = new_flanking.at[_C].set(
            semiring.plus(c_emitted, td_del_flanking[_C]))
        new_flanking = new_flanking.at[_JX].set(td_del_flanking[_JX])
        new_flanking = new_flanking.at[_J].set(
            semiring.plus(j_emitted, td_del_flanking[_J]))

        return new_core_m, new_core_i, new_core_d, new_flanking

    def _route_post_emission(core_m, core_i, core_d, flanking, semiring):
        """Route after emissions: Mx/Ix/D propagation + flanking silent.

        Inner scan over core positions for the M→next, D→next chains.
        Also handles insert self-loops and flanking routing.
        """
        # Mx_k routing from core_m[k]:
        #   → M_{k+1} (m_to_m), → I_k (m_to_i), → D_{k+1} (m_to_d), → E (local)
        # Ix_k routing from core_i[k]:
        #   → M_{k+1} (i_to_m), → I_k self (i_to_i), → E (local, last node only)
        # D_k routing from core_d[k]:
        #   → M_{k+1} (d_to_m), → D_{k+1} (d_to_d), → E (local)

        # Insert self-loop: I_k → Ix_k → I_k with weight i_to_i
        # Handle by geometric series: I_k_final = I_k / (1 - i_to_i) in prob space
        # In log space: I_k_final = I_k - log(1 - exp(i_to_i))... complex.
        # Better: just add the i_to_i contribution to core_i
        # Actually for the self-loop, Ix_k → I_k (i_to_i) means:
        # after I_k emits, goes to Ix_k, then Ix_k → I_k (self-loop).
        # This is handled in the NEXT emission step: core_i[k] will have
        # the i_to_i contribution added.

        # Inner scan: propagate through core left-to-right
        # At each k, compute contributions to k+1 and to E

        e_accum = jnp.full(S_td, NEG_INF)

        def inner_route(carry, k):
            m_incoming, d_incoming, e_acc = carry

            # Values at node k including incoming from left
            m_at_k = semiring.plus(core_m[k], m_incoming)
            i_at_k = core_i[k]  # I_k only from emissions, no left-propagation
            d_at_k = semiring.plus(core_d[k], d_incoming)

            # Propagate td silent
            m_at_k = _propagate_td_silent(m_at_k, td_silent, semiring)
            i_at_k = _propagate_td_silent(i_at_k, td_silent, semiring)
            d_at_k = _propagate_td_silent(d_at_k, td_silent, semiring)

            # E contributions (local mode)
            e_acc = semiring.plus(e_acc, m_at_k)
            e_acc = semiring.plus(e_acc, d_at_k)

            # Mx_k outgoing
            m_to_next = m_at_k + log_m_to_m[k]
            i_from_m = m_at_k + log_m_to_i[k]
            d_from_m = m_at_k + log_m_to_d[k]

            # Ix_k outgoing
            m_from_i = i_at_k + log_i_to_m[k]
            i_self = i_at_k + log_i_to_i[k]

            # D_k outgoing
            m_from_d = d_at_k + log_d_to_m[k]
            d_from_d = d_at_k + log_d_to_d[k]

            # Combined outgoing to next position
            m_out = semiring.plus(m_to_next, semiring.plus(m_from_i, m_from_d))
            d_out = semiring.plus(d_from_m, d_from_d)

            # I_k gets contribution from Mx_k → I_k and self-loop
            new_i_k = semiring.plus(i_from_m, i_self)

            return (m_out, d_out, e_acc), new_i_k

        init_carry = (jnp.full(S_td, NEG_INF), jnp.full(S_td, NEG_INF), e_accum)
        (m_exit, d_exit, e_final), new_core_i = jax.lax.scan(
            inner_route, init_carry, jnp.arange(K))

        # m_exit and d_exit flow to E (from last node, non-local transitions)
        # In local mode they already went to E inside the scan.
        # But for the last node (k=K-1 in 0-indexed), m_to_m and d_to_m
        # go to E in non-local mode. In local mode (which we use), they
        # already contributed to e_final inside the loop.
        # The m_exit/d_exit are the "overflow" past node K — discard them.

        # Flanking N emission routing: N → Nx (after emission)
        # N emitted already; now route through Nx
        new_flanking = jnp.full((N_FLANKING, S_td), NEG_INF)

        # Nx receives from N (post-emission → Nx)
        new_flanking = new_flanking.at[_NX].set(flanking[_N])
        # Note: flanking[_N] here is the POST-emission N value

        # Cx receives from C (post-emission → Cx)
        new_flanking = new_flanking.at[_CX].set(flanking[_C])

        # Jx receives from J (post-emission → Jx)
        new_flanking = new_flanking.at[_JX].set(flanking[_J])

        # E receives from core
        new_flanking = new_flanking.at[_E].set(e_final)

        # Now propagate silent flanking: Nx→N/B, E→Cx/Jx, Cx→C/T, Jx→J/B
        new_flanking = _propagate_flanking_silent(new_flanking, semiring)

        # B → core entry
        b_val = new_flanking[_B]
        new_core_m = log_b_entry[:, None] + b_val[None, :]  # (K, S_td)
        new_core_d = jnp.full((K, S_td), NEG_INF)

        return new_core_m, new_core_i, new_core_d, new_flanking

    def _get_terminal_val(flanking, semiring):
        """Get terminal state value: Cx → T."""
        t_val = flanking[_CX] + fm.log_c_to_t
        t_val = _propagate_td_silent(t_val, td_silent, semiring)
        return t_val[fm.S_td - 1]

    # Initialize
    core_m, core_i, core_d, flanking = _init_state()

    # Propagate initial silent transitions
    flanking = _propagate_flanking_silent(flanking, semiring)

    # B → core entry
    b_val = flanking[_B]
    core_m = log_b_entry[:, None] + b_val[None, :]

    # Handle Plan7 emissions that don't produce output (silent transducer output)
    # M_k and I_k can emit amino acids consumed silently by transducer
    def _emit_silent_core(core_m, core_i, semiring):
        """Plan7 core emits, transducer consumes silently (no output)."""
        def m_silent(k_data):
            m_val, m_emit_log = k_data
            return _td_emit_silent(m_val, m_emit_log, aa_to_td_in, td_log_trans, semiring)

        m_result = jax.vmap(m_silent)((core_m, log_match_emit))  # (K, S_td)

        def i_silent(k_data):
            i_val, i_emit_log = k_data
            return _td_emit_silent(i_val, i_emit_log, aa_to_td_in, td_log_trans, semiring)

        i_result = jax.vmap(i_silent)((core_i, log_ins_emit))  # (K, S_td)

        return semiring.plus(core_m, m_result), semiring.plus(core_i, i_result)

    # Initial: handle silent emissions + routing before first output
    core_m, core_i = _emit_silent_core(core_m, core_i, semiring)
    core_m, core_i, core_d, flanking = _route_post_emission(
        core_m, core_i, core_d, flanking, semiring)
    core_m, core_i = _emit_silent_core(core_m, core_i, semiring)

    if Lo == 0:
        return _get_terminal_val(flanking, semiring)

    # Outer scan over output positions
    def scan_fn(carry, out_tok):
        core_m, core_i, core_d, flanking = carry

        # Emit output
        core_m, core_i, core_d, flanking = _emit_output_step(
            core_m, core_i, core_d, flanking, out_tok, semiring)

        # Route and propagate
        core_m, core_i, core_d, flanking = _route_post_emission(
            core_m, core_i, core_d, flanking, semiring)

        # Silent Plan7 emissions (transducer consumes, no output)
        core_m, core_i = _emit_silent_core(core_m, core_i, semiring)

        return (core_m, core_i, core_d, flanking), None

    (core_m, core_i, core_d, flanking), _ = jax.lax.scan(
        scan_fn, (core_m, core_i, core_d, flanking), output_seq)

    return _get_terminal_val(flanking, semiring)


def fused_plan7_log_forward(fm: FusedPlan7Machine,
                            output_seq: jnp.ndarray) -> float:
    """Fused Plan7+transducer Forward algorithm with nested scans."""
    return _fused_plan7_dp(fm, output_seq, LOGSUMEXP)


def fused_plan7_log_viterbi(fm: FusedPlan7Machine,
                            output_seq: jnp.ndarray) -> float:
    """Fused Plan7+transducer Viterbi algorithm with nested scans."""
    return _fused_plan7_dp(fm, output_seq, MAXPLUS)
