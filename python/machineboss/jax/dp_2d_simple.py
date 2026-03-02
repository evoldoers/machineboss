"""2D DP engine using nested scans (SIMPLE strategy).

Dense: outer jax.lax.scan over rows, inner jax.lax.associative_scan
using augmented (S+1)x(S+1) matrices for the delete chain.
Sparse: outer jax.lax.scan + inner jax.lax.scan.
JIT-compilable: no Python for-loops in the DP computation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import JAXMachine, NEG_INF
from .semiring import LogSemiring, LOGSUMEXP, MAXPLUS
from .seq import TokenSeq, PSWMSeq, wrap_seq
from .kernel_dense import (
    propagate_silent, propagate_silent_backward,
    emit_step_forward, emit_step_backward,
)


def _precompute_emit_trans(emit, trans_slice, semiring):
    """Precompute weighted transition matrices for all positions.

    Args:
        emit: (L, n_tokens) emission weights
        trans_slice: (n_tokens, S, S) transition matrices
        semiring: LogSemiring
    Returns:
        (L, S, S) per-position transition matrices.
    """
    def make_one(e):
        e = e.at[0].set(NEG_INF)
        return semiring.reduce(e[:, None, None] + trans_slice, axis=0)
    return jax.vmap(make_one)(emit)


def _precompute_match_trans_row(in_e, out_emit, log_trans, semiring):
    """Precompute match transition matrices for one input row, all output positions.

    Args:
        in_e: (n_in,) input emission weights for this row
        out_emit: (Lo, n_out) output emission weights
        log_trans: (n_in, n_out, S, S) full transition tensor
        semiring: LogSemiring
    Returns:
        (Lo, S, S) match transition matrices.
    """
    in_e_clean = in_e.at[0].set(NEG_INF)
    # First reduce over input tokens: in_marginal[out, S, S]
    in_marginal = semiring.reduce(
        in_e_clean[:, None, None, None] + log_trans, axis=0)  # (n_out, S, S)

    def make_match(out_e):
        out_e_clean = out_e.at[0].set(NEG_INF)
        return semiring.reduce(out_e_clean[:, None, None] + in_marginal, axis=0)

    return jax.vmap(make_match)(out_emit)  # (Lo, S, S)


def _silent_closure(silent, semiring, max_iter=100):
    """Compute Kleene star of silent transition matrix."""
    S = silent.shape[0]
    identity = jnp.full((S, S), NEG_INF)
    identity = identity.at[jnp.arange(S), jnp.arange(S)].set(0.0)

    def body_fn(carry):
        prev_result, power, _ = carry
        next_power = semiring.mat_mul(power, silent)
        new_result = semiring.plus(prev_result, next_power)
        return new_result, next_power, prev_result

    def cond_fn(carry):
        new_result, _, prev_result = carry
        return jnp.any(jnp.abs(new_result - prev_result) > 1e-10)

    init = (identity, identity, jnp.full_like(identity, NEG_INF))
    result, _, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return result


def _matvec_fwd(trans, v, semiring):
    """Matrix-vector product: result[dst] = reduce_src(v[src] + trans[src, dst])."""
    return semiring.reduce(v[:, None] + trans, axis=0)


# ============================================================
# Dense Forward
# ============================================================

def forward_2d_dense(machine: JAXMachine, input_seq, output_seq,
                     semiring: LogSemiring) -> float:
    """2D Forward/Viterbi using outer scan + inner associative scan."""
    assert machine.log_trans is not None
    S = machine.n_states
    log_trans = machine.log_trans

    input_seq = wrap_seq(input_seq, machine.n_input_tokens)
    output_seq = wrap_seq(output_seq, machine.n_output_tokens)

    Li = len(input_seq) if input_seq is not None else 0
    Lo = len(output_seq) if output_seq is not None else 0

    in_emit = input_seq.emission_weights(machine.n_input_tokens) if Li > 0 else None
    out_emit = output_seq.emission_weights(machine.n_output_tokens) if Lo > 0 else None

    silent = log_trans[0, 0]
    closure = _silent_closure(silent, semiring)

    # Precompute delete transition matrices: del_trans[o] for o = 0..Lo-1
    if Lo > 0:
        del_trans = _precompute_emit_trans(
            out_emit, log_trans[0, :, :, :], semiring)  # (Lo, S, S)
        # A_del[o] = del_trans[o] @ closure (delete then silent propagation)
        A_del = jax.vmap(lambda d: semiring.mat_mul(d, closure))(del_trans)  # (Lo, S, S)

    # Precompute insert transition matrices: ins_trans[i] for i = 0..Li-1
    if Li > 0:
        ins_trans = _precompute_emit_trans(
            in_emit, log_trans[:, 0, :, :], semiring)  # (Li, S, S)
        # A_ins_closure[i] = ins_trans[i] @ closure
        ins_closure = jax.vmap(lambda t: semiring.mat_mul(t, closure))(ins_trans)

    # --- Compute row 0: only delete transitions ---
    row0 = jnp.full((Lo + 1, S), NEG_INF)
    row0 = row0.at[0, 0].set(0.0)
    row0 = row0.at[0].set(_matvec_fwd(closure, row0[0], semiring))

    if Lo > 0:
        # Row 0 delete chain: row0[o] = A_del[o-1] @ row0[o-1]
        def scan_row0(cell, a):
            return _matvec_fwd(a, cell, semiring), None
        row0_final, _ = jax.lax.scan(scan_row0, row0[0], A_del)
        # Actually need all intermediate values too
        def scan_row0_collect(cell, a):
            new = _matvec_fwd(a, cell, semiring)
            return new, new
        _, row0_del = jax.lax.scan(scan_row0_collect, row0[0], A_del)
        row0 = jnp.concatenate([row0[0:1], row0_del], axis=0)

    if Li == 0:
        return row0[Lo, S - 1]

    # --- Outer scan over rows 1..Li ---
    S1 = S + 1  # augmented matrix size

    def _process_row(prev_row, row_idx):
        """Compute one row given the previous row."""
        # Get insert/match trans for this row
        this_ins = ins_trans[row_idx]  # (S, S)
        this_ins_closure = ins_closure[row_idx]  # (S, S)

        # dp[i, 0] = closure @ (ins_trans @ prev_row[0])
        cell_0 = _matvec_fwd(this_ins_closure, prev_row[0], semiring)

        if Lo == 0:
            return jnp.concatenate([cell_0[None, :]], axis=0), None

        # Match transitions for all output positions
        this_in_e = in_emit[row_idx]  # (n_in,)
        match_trans = _precompute_match_trans_row(
            this_in_e, out_emit, log_trans, semiring)  # (Lo, S, S)
        # match_closure[o] = match_trans[o] @ closure
        match_closure = jax.vmap(
            lambda m: semiring.mat_mul(m, closure))(match_trans)  # (Lo, S, S)

        # Free inputs for o = 1..Lo (index k = 0..Lo-1):
        # c_match[k] = matvec(match_closure[k], prev_row[k])
        # c_insert[k] = matvec(ins_closure, prev_row[k+1])
        # c[k] = c_match[k] ⊕ c_insert[k]
        c_match = jax.vmap(
            lambda t, v: _matvec_fwd(t, v, semiring))(match_closure, prev_row[:Lo])
        c_insert = jax.vmap(
            lambda v: _matvec_fwd(this_ins_closure, v, semiring))(prev_row[1:Lo+1])
        c = semiring.plus(c_match, c_insert)  # (Lo, S)

        # Build augmented matrices for inner associative scan (row-vector convention).
        # _matvec_fwd computes v @ M, so the augmented matrix is:
        # M[k] = [[A_del[k], NEG_INF_col], [c[k], 0]]
        # Then [v, 0] @ M = [v @ A_del ⊕ c, 0].
        def build_aug(a, ci):
            M = jnp.full((S1, S1), NEG_INF)
            M = M.at[:S, :S].set(a)
            M = M.at[S, :S].set(ci)
            M = M.at[S, S].set(0.0)
            return M

        augs = jax.vmap(build_aug)(A_del, c)  # (Lo, S1, S1)

        # Prefix product via associative scan
        prefix = jax.lax.associative_scan(
            lambda a, b: semiring.mat_mul(a, b),
            augs, axis=0)  # (Lo, S1, S1)

        # Extract: [v, 0] @ prefix[k] gives [v @ A_acc ⊕ c_acc, 0]
        def extract(p):
            Av = _matvec_fwd(p[:S, :S], cell_0, semiring)
            return semiring.plus(Av, p[S, :S])

        cells = jax.vmap(extract)(prefix)  # (Lo, S)
        new_row = jnp.concatenate([cell_0[None, :], cells], axis=0)  # (Lo+1, S)
        return new_row, None

    # Scan over input positions
    final_row, _ = jax.lax.scan(_process_row, row0, jnp.arange(Li))
    return final_row[Lo, S - 1]


# ============================================================
# Dense Backward
# ============================================================

def backward_2d_dense(machine: JAXMachine, input_seq, output_seq,
                      semiring: LogSemiring) -> jnp.ndarray:
    """2D Backward using outer scan + inner associative scan."""
    assert machine.log_trans is not None
    S = machine.n_states
    log_trans = machine.log_trans

    input_seq = wrap_seq(input_seq, machine.n_input_tokens)
    output_seq = wrap_seq(output_seq, machine.n_output_tokens)

    Li = len(input_seq) if input_seq is not None else 0
    Lo = len(output_seq) if output_seq is not None else 0

    in_emit = input_seq.emission_weights(machine.n_input_tokens) if Li > 0 else None
    out_emit = output_seq.emission_weights(machine.n_output_tokens) if Lo > 0 else None

    silent = log_trans[0, 0]
    closure = _silent_closure(silent, semiring)
    # Backward closure: closure_bwd[src, dst] = closure[dst, src]... no.
    # For backward: result[src] = reduce_dst(trans[src,dst] + future[dst])
    # This means: result = matvec_bwd(trans, future) where
    # matvec_bwd(trans, v)[src] = reduce_dst(trans[src,dst] + v[dst])
    # = reduce_dst(v[dst] + trans[src,dst])
    # In matrix terms with trans[src,dst]: result = trans @ v (standard multiply)

    # Precompute delete transition matrices (backward: delete emits output[o] going to o+1)
    if Lo > 0:
        del_trans = _precompute_emit_trans(
            out_emit, log_trans[0, :, :, :], semiring)  # (Lo, S, S)

    if Li > 0:
        ins_trans = _precompute_emit_trans(
            in_emit, log_trans[:, 0, :, :], semiring)  # (Li, S, S)

    # Terminal cell
    bp_term = jnp.full((Lo + 1, S), NEG_INF)
    bp_term = bp_term.at[Lo, S - 1].set(0.0)
    # Propagate silent backward at terminal position
    bp_term = bp_term.at[Lo].set(
        _matvec_bwd(closure, bp_term[Lo], semiring))

    # Compute last row (i = Li): only delete transitions, backward
    if Lo > 0:
        def scan_last_row(cell, inputs):
            dt = inputs  # del_trans for this output position
            contrib = _matvec_bwd(dt, cell, semiring)
            new_cell = _matvec_bwd(closure, contrib, semiring)
            return new_cell, new_cell

        # Process columns right to left: Lo-1, Lo-2, ..., 0
        _, bp_last_rev = jax.lax.scan(
            scan_last_row, bp_term[Lo], del_trans[::-1])
        bp_last_cells = bp_last_rev[::-1]  # (Lo, S)
        last_row = jnp.concatenate([bp_last_cells, bp_term[Lo:Lo+1]], axis=0)
    else:
        last_row = bp_term

    if Li == 0:
        return last_row[None, :, :] if last_row.ndim == 2 else last_row

    # Collect all rows: we need to scan backward over i
    S1 = S + 1

    def _process_row_bwd(next_row, row_idx):
        """Compute backward values for row i given row i+1."""
        this_ins = ins_trans[row_idx]  # (S, S)

        # bp[i, Lo] has no outgoing transitions (already at last output position)
        # Only insert: bp[i, Lo] += matvec_bwd(ins_trans, bp[i+1, Lo])
        cell_Lo = _matvec_bwd(this_ins, next_row[Lo], semiring)
        cell_Lo = _matvec_bwd(closure, cell_Lo, semiring)

        if Lo == 0:
            return cell_Lo[None, :], cell_Lo[None, :]

        # Match transitions for all output positions
        this_in_e = in_emit[row_idx]
        match_trans = _precompute_match_trans_row(
            this_in_e, out_emit, log_trans, semiring)

        # For backward at (i, o), transitions go to:
        #   match: (i+1, o+1) via match_trans[o]
        #   insert: (i+1, o) via ins_trans
        #   delete: (i, o+1) via del_trans[o]
        # The delete chain goes right-to-left: bp[i,o] depends on bp[i,o+1]

        # Free inputs for o = Lo-1, Lo-2, ..., 0 (index k = 0..Lo-1 from right):
        # At position o:
        #   match_input = matvec_bwd(match_trans[o], next_row[o+1])
        #   insert_input = matvec_bwd(ins_trans, next_row[o])
        # We process right to left, so k=0 is o=Lo-1, k=1 is o=Lo-2, etc.

        # Compute match contributions: match at (i,o) uses next_row[o+1]
        c_match = jax.vmap(
            lambda mt, future: _matvec_bwd(mt, future, semiring)
        )(match_trans, next_row[1:Lo+1])  # (Lo, S)

        # Compute insert contributions: insert at (i,o) uses next_row[o]
        c_insert = jax.vmap(
            lambda future: _matvec_bwd(this_ins, future, semiring)
        )(next_row[:Lo])  # (Lo, S)

        c = semiring.plus(c_match, c_insert)  # (Lo, S)

        # Delete chain (right to left): bp[i,o] depends on bp[i,o+1] via del_trans[o]
        # Recurrence: bp[i,o] = closure_bwd @ (del_trans[o]_bwd @ bp[i,o+1] ⊕ c[o])
        # A_del_bwd[o] = closure_bwd(del_trans[o]_bwd) where
        # _bwd means backward application

        # For backward delete: matvec_bwd(del_trans[o], v)[src] = reduce_dst(del_trans[o][src,dst] + v[dst])
        # Combined with closure: A[o] = semiring.mat_mul(closure^T, del_trans[o]) ... hmm

        # Actually, for backward the matrix chain is:
        # bp[i,o] = reduce_dst(closure[src,dst] * (reduce_dst2(del_trans[o][dst, dst2] * bp[i,o+1][dst2]) + c[o][dst]))
        # This is: bp[i,o] = closure @ (del_trans[o] @ bp[i,o+1] + c[o]) where @ is standard matvec

        # Wait, let me use the semiring notation more carefully.
        # For backward at (i, o), the delete transition goes from (i, o) to (i, o+1):
        # The backward value contribution is:
        # update[src] = reduce_dst(trans[src,dst] + bp[i,o+1][dst])
        # Then: bp[i,o][src] += update[src]

        # After all contributions:
        # bp[i,o] = silent_prop_bwd(match_bwd + insert_bwd + delete_bwd)
        # where silent_prop_bwd applies closure in backward direction:
        # result[src] = reduce_dst(closure[src,dst] + cell[dst])

        # Wait, silent_prop_bwd: silent transitions are [src,dst], backward propagation:
        # result[src] = reduce_dst(silent[src,dst] + cell[dst])
        # Closure_bwd[src] = reduce_dst(closure[src,dst] + cell[dst])

        # So the delete chain is:
        # bp[i,o] = closure_bwd(del_bwd(bp[i,o+1]) + c[o])
        # = reduce_d(closure[s,d] * (reduce_d2(del_trans[o][d,d2] * bp[i,o+1][d2]) + c[o][d]))

        # In matrix notation: bp[i,o] = closure @ (del_trans[o] @ bp[i,o+1] + c[o])
        # where @ is standard matrix multiply and + is logaddexp.

        # Since closure distributes:
        # bp[i,o] = (closure @ del_trans[o]) @ bp[i,o+1] + closure @ c[o]

        # Let A[o] = closure @ del_trans[o], b[o] = closure @ c[o]
        # Recurrence: bp[i,o] = A[o] @ bp[i,o+1] + b[o]

        # This is the same type of affine recurrence! Process right-to-left.

        # But we need to be careful about the matrix multiply convention.
        # "closure @ del_trans[o]" in semiring: C[i,j] = reduce_k(closure[i,k] + del_trans[o][k,j])
        # This is semiring.mat_mul(closure, del_trans[o])

        # "A @ v" (matvec): result[i] = reduce_j(v[j] + A[i,j])
        # Wait no, let me re-check _matvec_bwd:
        # _matvec_bwd(trans, v)[src] = reduce_dst(trans[src,dst] + v[dst])

        # So (A @ v)[src] = reduce_dst(A[src,dst] + v[dst]) = _matvec_bwd(A, v)

        # Composing: _matvec_bwd(A, _matvec_bwd(del, v))
        # = _matvec_bwd(A, result1) where result1[d] = reduce_d2(del[d,d2]+v[d2])
        # = reduce_d(A[s,d] + result1[d])
        # = reduce_d(A[s,d] + reduce_d2(del[d,d2]+v[d2]))
        # = reduce_{d,d2}(A[s,d] + del[d,d2] + v[d2])  (by distributivity)
        # = reduce_d2(reduce_d(A[s,d] + del[d,d2]) + v[d2])
        # = _matvec_bwd(A@del, v)
        # where (A@del)[s,d2] = reduce_d(A[s,d] + del[d,d2]) = semiring.mat_mul(A, del)

        # So the combined matrix is: closure @ del_trans = semiring.mat_mul(closure, del_trans[o])
        A_del_bwd = jax.vmap(
            lambda d: semiring.mat_mul(closure, d))(del_trans)  # (Lo, S, S)

        c_closed = jax.vmap(
            lambda ci: _matvec_bwd(closure, ci, semiring))(c)  # (Lo, S)

        # Build augmented matrices for right-to-left scan (column-vector convention).
        # _matvec_bwd computes M @ v, so the augmented matrix is:
        # M[k] = [[A[k], c[k]], [NEG_INF_row, 0]]
        # Then M @ [v; 1] = [A@v ⊕ c; 1].
        # Reverse the arrays: process from o = Lo-1 down to 0.
        def build_aug(a, ci):
            M = jnp.full((S1, S1), NEG_INF)
            M = M.at[:S, :S].set(a)
            M = M.at[:S, S].set(ci)
            M = M.at[S, S].set(0.0)
            return M

        augs = jax.vmap(build_aug)(A_del_bwd[::-1], c_closed[::-1])

        # For column vectors, we need prefix[k] = M[k] @ ... @ M[0],
        # so use reverse argument order in the scan.
        prefix = jax.lax.associative_scan(
            lambda a, b: semiring.mat_mul(b, a),
            augs, axis=0)

        # Extract: prefix[k] @ [cell_Lo; 1] gives [A_acc @ cell_Lo ⊕ c_acc; 1]
        def extract(p):
            Av = _matvec_bwd(p[:S, :S], cell_Lo, semiring)
            return semiring.plus(Av, p[:S, S])

        cells_rev = jax.vmap(extract)(prefix)  # (Lo, S)
        cells = cells_rev[::-1]  # unreverse: cells[o] for o = 0..Lo-1

        new_row = jnp.concatenate([cells, cell_Lo[None, :]], axis=0)  # (Lo+1, S)
        return new_row, new_row

    # Scan backward over rows: i = Li-1, Li-2, ..., 0
    _, bp_rows_rev = jax.lax.scan(
        _process_row_bwd, last_row, jnp.arange(Li)[::-1])
    bp_rows = bp_rows_rev[::-1]  # (Li, Lo+1, S)

    # Full backward matrix: (Li+1, Lo+1, S)
    bp = jnp.concatenate([bp_rows, last_row[None, :, :]], axis=0)
    return bp


def _matvec_bwd(trans, v, semiring):
    """Backward matvec: result[src] = reduce_dst(trans[src,dst] + v[dst])."""
    return semiring.reduce(trans + v[None, :], axis=1)


# ============================================================
# Sparse Forward
# ============================================================

def forward_2d_sparse(machine: JAXMachine, input_seq, output_seq,
                      semiring: LogSemiring) -> float:
    """2D Forward/Viterbi using sparse COO and nested jax.lax.scan."""
    from .kernel_sparse import (
        propagate_silent_sparse, emit_step_forward_sparse_pswm,
    )

    S = machine.n_states

    input_seq = wrap_seq(input_seq, machine.n_input_tokens)
    output_seq = wrap_seq(output_seq, machine.n_output_tokens)

    Li = len(input_seq) if input_seq is not None else 0
    Lo = len(output_seq) if output_seq is not None else 0

    in_emit = input_seq.emission_weights(machine.n_input_tokens) if Li > 0 else None
    out_emit = output_seq.emission_weights(machine.n_output_tokens) if Lo > 0 else None

    # --- Row 0: only delete transitions ---
    init_cell = jnp.full(S, NEG_INF).at[0].set(0.0)
    init_cell = propagate_silent_sparse(init_cell, machine, semiring)

    if Lo > 0:
        def scan_row0_del(cell, out_e):
            new = emit_step_forward_sparse_pswm(
                jnp.full(S, NEG_INF), cell, machine,
                None, out_e, emit_in=False, emit_out=True, semiring=semiring)
            new = propagate_silent_sparse(new, machine, semiring)
            return new, new
        _, row0_cells = jax.lax.scan(scan_row0_del, init_cell, out_emit)
        row0 = jnp.concatenate([init_cell[None, :], row0_cells], axis=0)
    else:
        row0 = init_cell[None, :]

    if Li == 0:
        return row0[Lo, S - 1]

    # --- Outer scan over rows ---
    def _process_row_sparse(prev_row, in_e):
        # Inner scan over columns
        def inner_step(cell_left, col_data):
            o_idx, out_e, prev_match, prev_ins = col_data
            cell = jnp.full(S, NEG_INF)

            # Match: from prev_row[o-1] (= prev_match)
            cell = emit_step_forward_sparse_pswm(
                cell, prev_match, machine,
                in_e, out_e, emit_in=True, emit_out=True, semiring=semiring)

            # Insert: from prev_row[o] (= prev_ins)
            cell = emit_step_forward_sparse_pswm(
                cell, prev_ins, machine,
                in_e, None, emit_in=True, emit_out=False, semiring=semiring)

            # Delete: from cell_left
            cell = emit_step_forward_sparse_pswm(
                cell, cell_left, machine,
                None, out_e, emit_in=False, emit_out=True, semiring=semiring)

            cell = propagate_silent_sparse(cell, machine, semiring)
            return cell, cell

        # Column 0: insert only
        cell_0 = emit_step_forward_sparse_pswm(
            jnp.full(S, NEG_INF), prev_row[0], machine,
            in_e, None, emit_in=True, emit_out=False, semiring=semiring)
        cell_0 = propagate_silent_sparse(cell_0, machine, semiring)

        if Lo == 0:
            return cell_0[None, :], None

        # Columns 1..Lo
        col_indices = jnp.arange(Lo)
        prev_match_cells = prev_row[:Lo]  # prev_row[0..Lo-1] for match at o=1..Lo
        prev_ins_cells = prev_row[1:Lo+1]  # prev_row[1..Lo] for insert at o=1..Lo

        _, inner_cells = jax.lax.scan(
            inner_step, cell_0,
            (col_indices, out_emit, prev_match_cells, prev_ins_cells))

        new_row = jnp.concatenate([cell_0[None, :], inner_cells], axis=0)
        return new_row, None

    final_row, _ = jax.lax.scan(_process_row_sparse, row0, in_emit)
    return final_row[Lo, S - 1]


# ============================================================
# Sparse Backward
# ============================================================

def backward_2d_sparse(machine: JAXMachine, input_seq, output_seq,
                       semiring: LogSemiring) -> jnp.ndarray:
    """2D Backward using sparse COO and nested jax.lax.scan."""
    from .kernel_sparse import (
        propagate_silent_backward_sparse, emit_step_backward_sparse_pswm,
    )

    S = machine.n_states

    input_seq = wrap_seq(input_seq, machine.n_input_tokens)
    output_seq = wrap_seq(output_seq, machine.n_output_tokens)

    Li = len(input_seq) if input_seq is not None else 0
    Lo = len(output_seq) if output_seq is not None else 0

    in_emit = input_seq.emission_weights(machine.n_input_tokens) if Li > 0 else None
    out_emit = output_seq.emission_weights(machine.n_output_tokens) if Lo > 0 else None

    # Terminal
    term = jnp.full(S, NEG_INF).at[S - 1].set(0.0)
    term = propagate_silent_backward_sparse(term, machine, semiring)

    # Last row (i = Li): only delete transitions, right to left
    if Lo > 0:
        def scan_last_del(cell, out_e):
            new = emit_step_backward_sparse_pswm(
                jnp.full(S, NEG_INF), cell, machine,
                None, out_e, emit_in=False, emit_out=True, semiring=semiring)
            new = propagate_silent_backward_sparse(new, machine, semiring)
            return new, new
        _, last_rev = jax.lax.scan(scan_last_del, term, out_emit[::-1])
        last_cells = last_rev[::-1]
        last_row = jnp.concatenate([last_cells, term[None, :]], axis=0)
    else:
        last_row = term[None, :]

    if Li == 0:
        return last_row[None, :, :] if last_row.ndim == 2 else last_row

    # Backward outer scan over rows
    def _process_row_bwd_sparse(next_row, in_e):
        # Column Lo: insert only
        cell_Lo = emit_step_backward_sparse_pswm(
            jnp.full(S, NEG_INF), next_row[Lo], machine,
            in_e, None, emit_in=True, emit_out=False, semiring=semiring)
        cell_Lo = propagate_silent_backward_sparse(cell_Lo, machine, semiring)

        if Lo == 0:
            return cell_Lo[None, :], cell_Lo[None, :]

        # Inner scan right-to-left
        def inner_step(cell_right, col_data):
            out_e, next_match, next_ins = col_data
            cell = jnp.full(S, NEG_INF)

            # Match: to (i+1, o+1) = next_row[o+1] = next_match
            cell = emit_step_backward_sparse_pswm(
                cell, next_match, machine,
                in_e, out_e, emit_in=True, emit_out=True, semiring=semiring)

            # Insert: to (i+1, o) = next_row[o] = next_ins
            cell = emit_step_backward_sparse_pswm(
                cell, next_ins, machine,
                in_e, None, emit_in=True, emit_out=False, semiring=semiring)

            # Delete: to (i, o+1) = cell_right
            cell = emit_step_backward_sparse_pswm(
                cell, cell_right, machine,
                None, out_e, emit_in=False, emit_out=True, semiring=semiring)

            cell = propagate_silent_backward_sparse(cell, machine, semiring)
            return cell, cell

        # Process columns Lo-1, Lo-2, ..., 0
        next_match_cells = next_row[1:Lo+1][::-1]  # next_row[Lo..1] reversed
        next_ins_cells = next_row[:Lo][::-1]  # next_row[Lo-1..0] reversed

        _, inner_rev = jax.lax.scan(
            inner_step, cell_Lo,
            (out_emit[::-1], next_match_cells, next_ins_cells))
        inner_cells = inner_rev[::-1]  # (Lo, S)

        new_row = jnp.concatenate([inner_cells, cell_Lo[None, :]], axis=0)
        return new_row, new_row

    # Scan backward over rows
    _, bp_rows_rev = jax.lax.scan(
        _process_row_bwd_sparse, last_row, in_emit[::-1])
    bp_rows = bp_rows_rev[::-1]  # (Li, Lo+1, S)

    bp = jnp.concatenate([bp_rows, last_row[None, :, :]], axis=0)
    return bp
