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
    #   core_m: (K, S_td) — at M_k pre-emission (ready for next emission)
    #   core_i: (K, S_td) — at I_k pre-emission
    #   core_d: (K, S_td) — at D_k (silent)
    #   flanking: (N_FLANKING, S_td)

    def _propagate_flanking_silent(flanking, semiring):
        """Propagate silent flanking transitions (single-pass DAG).

        The flanking silent graph is acyclic:
          E->CX, E->JX, NX->B, JX->B, NX->N, CX->C, JX->J
        A single pass in topological order suffices.
        After applying flanking edges, propagate transducer silent at each state.
        """
        fl = flanking

        # E -> CX
        fl = fl.at[_CX].set(semiring.plus(fl[_CX], fl[_E] + fm.log_e_to_cx))

        # E -> JX
        fl = fl.at[_JX].set(semiring.plus(fl[_JX], fl[_E] + fm.log_e_to_jx))

        # NX -> B
        fl = fl.at[_B].set(semiring.plus(fl[_B], fl[_NX] + fm.log_n_to_b))

        # JX -> B (must come after E -> JX)
        fl = fl.at[_B].set(semiring.plus(fl[_B], fl[_JX] + fm.log_j_to_b))

        # NX -> N
        fl = fl.at[_N].set(semiring.plus(fl[_N], fl[_NX] + fm.log_n_loop))

        # CX -> C (must come after E -> CX)
        fl = fl.at[_C].set(semiring.plus(fl[_C], fl[_CX] + fm.log_c_loop))

        # JX -> J (must come after E -> JX)
        fl = fl.at[_J].set(semiring.plus(fl[_J], fl[_JX] + fm.log_j_loop))

        # Propagate td silent at each flanking state
        fl = jax.vmap(
            lambda v: _propagate_td_silent(v, td_silent, semiring)
        )(fl)

        return fl

    def _emit_output_step(core_m, core_i, core_d, flanking, out_tok, semiring):
        """Process one output token: Plan7 emits -> transducer produces output.

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

        Inner scan over core positions for the M->next, D->next chains.
        Also handles insert self-loops and flanking routing.

        Key invariant: core_m[k] is POST-emission (at Mx_k). The inner scan's
        m_incoming carries PRE-emission mass arriving at M_k from left routing.
        Only pre-emission M_k and D_k contribute to E (local exit).
        Only post-emission Mx_k is routed through m_to_m/i/d.
        Pre-emission M_k values persist in the returned core_m for next emission.
        """
        e_accum = jnp.full(S_td, NEG_INF)

        def inner_route(carry, k):
            m_incoming, d_incoming, e_acc = carry

            # D_k: combine emit-step D_k with incoming from left
            d_at_k = semiring.plus(core_d[k], d_incoming)
            d_at_k = _propagate_td_silent(d_at_k, td_silent, semiring)

            # Pre-emission M_k from routing (ONLY m_incoming, not core_m[k])
            m_pre_k = _propagate_td_silent(m_incoming, td_silent, semiring)

            # Post-emission Mx_k (core_m[k])
            mx_k = _propagate_td_silent(core_m[k], td_silent, semiring)

            # Post-emission Ix_k (core_i[k])
            i_at_k = _propagate_td_silent(core_i[k], td_silent, semiring)

            # E contributions: ONLY pre-emission M_k and D_k (not Mx_k)
            e_acc = semiring.plus(e_acc, m_pre_k)
            e_acc = semiring.plus(e_acc, d_at_k)

            # Mx_k outgoing (post-emission only)
            m_to_next = mx_k + log_m_to_m[k]
            i_from_m = mx_k + log_m_to_i[k]
            d_from_m = mx_k + log_m_to_d[k]

            # Ix_k outgoing
            m_from_i = i_at_k + log_i_to_m[k]
            i_self = i_at_k + log_i_to_i[k]

            # D_k outgoing
            m_from_d = d_at_k + log_d_to_m[k]
            d_from_d = d_at_k + log_d_to_d[k]

            # Combined outgoing to next position
            m_out = semiring.plus(m_to_next, semiring.plus(m_from_i, m_from_d))
            d_out = semiring.plus(d_from_m, d_from_d)

            # I_k gets contribution from Mx_k -> I_k and self-loop
            new_i_k = semiring.plus(i_from_m, i_self)

            return (m_out, d_out, e_acc), (m_pre_k, new_i_k)

        init_carry = (jnp.full(S_td, NEG_INF), jnp.full(S_td, NEG_INF), e_accum)
        (m_exit, d_exit, e_final), (all_m_pre, new_core_i) = jax.lax.scan(
            inner_route, init_carry, jnp.arange(K))

        # Build new flanking
        new_flanking = jnp.full((N_FLANKING, S_td), NEG_INF)

        # Nx receives from N (post-emission -> Nx)
        new_flanking = new_flanking.at[_NX].set(flanking[_N])

        # Cx receives from C (post-emission -> Cx)
        new_flanking = new_flanking.at[_CX].set(flanking[_C])

        # Jx receives from J (post-emission -> Jx)
        new_flanking = new_flanking.at[_JX].set(flanking[_J])

        # E receives from core
        new_flanking = new_flanking.at[_E].set(e_final)

        # Propagate silent flanking: NX->N/B, E->CX/JX, CX->C, JX->J/B
        new_flanking = _propagate_flanking_silent(new_flanking, semiring)

        # Final core_m = pre-emission from routing + new B entry
        b_val = new_flanking[_B]
        new_core_m = semiring.plus(
            all_m_pre, log_b_entry[:, None] + b_val[None, :])

        # B -> M_k -> E chain: new B-entry M_k can immediately exit to E
        b_entries = log_b_entry[:, None] + b_val[None, :]  # (K, S_td)
        b_entries_closed = jax.vmap(
            lambda v: _propagate_td_silent(v, td_silent, semiring)
        )(b_entries)  # (K, S_td)
        e_from_b = semiring.reduce(b_entries_closed, axis=0)  # (S_td,)

        # Propagate E -> CX -> C and E -> JX -> J -> B
        e_closed = _propagate_td_silent(e_from_b, td_silent, semiring)

        new_flanking = new_flanking.at[_E].set(
            semiring.plus(new_flanking[_E], e_from_b))

        # E -> CX
        cx_inc = e_closed + fm.log_e_to_cx
        new_flanking = new_flanking.at[_CX].set(
            semiring.plus(new_flanking[_CX], cx_inc))

        # CX -> C
        c_inc = cx_inc + fm.log_c_loop
        new_flanking = new_flanking.at[_C].set(
            semiring.plus(new_flanking[_C], c_inc))

        # E -> JX
        jx_inc = e_closed + fm.log_e_to_jx
        new_flanking = new_flanking.at[_JX].set(
            semiring.plus(new_flanking[_JX], jx_inc))

        # JX -> J
        j_inc = jx_inc + fm.log_j_loop
        new_flanking = new_flanking.at[_J].set(
            semiring.plus(new_flanking[_J], j_inc))

        # JX -> B -> M_k (multi-hit)
        b_inc = jx_inc + fm.log_j_to_b
        new_flanking = new_flanking.at[_B].set(
            semiring.plus(new_flanking[_B], b_inc))
        new_core_m = semiring.plus(
            new_core_m, log_b_entry[:, None] + b_inc[None, :])

        new_core_d = jnp.full((K, S_td), NEG_INF)

        return new_core_m, new_core_i, new_core_d, new_flanking

    def _get_terminal_val(flanking, semiring):
        """Get terminal state value: Cx -> T."""
        t_val = flanking[_CX] + fm.log_c_to_t
        t_val = _propagate_td_silent(t_val, td_silent, semiring)
        return t_val[fm.S_td - 1]

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

    # Initialize: manual 8-step DAG propagation (matching JS kernel)
    core_m = jnp.full((K, S_td), NEG_INF)
    core_i = jnp.full((K, S_td), NEG_INF)
    core_d = jnp.full((K, S_td), NEG_INF)
    flanking = jnp.full((N_FLANKING, S_td), NEG_INF)

    # Step 1: S -> NX; propagate td_silent at NX
    flanking = flanking.at[_NX, 0].set(0.0)
    flanking = flanking.at[_NX].set(
        _propagate_td_silent(flanking[_NX], td_silent, semiring))

    # Step 2: NX -> B, NX -> N; propagate td_silent at B, N
    flanking = flanking.at[_B].set(flanking[_NX] + fm.log_n_to_b)
    flanking = flanking.at[_N].set(flanking[_NX] + fm.log_n_loop)
    flanking = flanking.at[_B].set(
        _propagate_td_silent(flanking[_B], td_silent, semiring))
    flanking = flanking.at[_N].set(
        _propagate_td_silent(flanking[_N], td_silent, semiring))

    # Step 3: B -> M_k (pre-emission entry)
    b_val = flanking[_B]
    core_m = log_b_entry[:, None] + b_val[None, :]

    # Step 4: M_k -> E (pre-emission local exit, weight 1); propagate td_silent at E
    m_closed = jax.vmap(
        lambda v: _propagate_td_silent(v, td_silent, semiring)
    )(core_m)
    e_val = semiring.reduce(m_closed, axis=0)
    flanking = flanking.at[_E].set(e_val)
    flanking = flanking.at[_E].set(
        _propagate_td_silent(flanking[_E], td_silent, semiring))

    # Step 5: E -> CX, E -> JX
    flanking = flanking.at[_CX].set(flanking[_E] + fm.log_e_to_cx)
    flanking = flanking.at[_JX].set(flanking[_E] + fm.log_e_to_jx)

    # Step 6: JX -> B (multi-hit)
    flanking = flanking.at[_B].set(
        semiring.plus(flanking[_B], flanking[_JX] + fm.log_j_to_b))

    # Step 7: CX -> C, JX -> J; propagate td_silent at CX, JX, C, J
    flanking = flanking.at[_C].set(flanking[_CX] + fm.log_c_loop)
    flanking = flanking.at[_J].set(flanking[_JX] + fm.log_j_loop)
    for f in [_CX, _JX, _C, _J]:
        flanking = flanking.at[f].set(
            _propagate_td_silent(flanking[f], td_silent, semiring))

    # Step 8: Multi-hit B -> M_k entries (from JX -> B increment)
    jx_to_b = flanking[_JX] + fm.log_j_to_b
    b_inc = _propagate_td_silent(jx_to_b, td_silent, semiring)
    core_m = semiring.plus(core_m, log_b_entry[:, None] + b_inc[None, :])

    # Handle initial silent emissions
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
