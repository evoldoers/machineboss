//! Fused Plan7 + transducer Forward/Viterbi kernel.
//!
//! Faithful Rust port of `js/webgpu/cpu/fused-plan7.mjs`. A 1-D DP over output
//! positions, exploiting the linear-chain structure of a Plan7 profile HMM so
//! the composed Plan7 x transducer state space is never materialized.
//!
//! The ONLY intentional deviation from the JS reference is that the transducer
//! transitions are iterated **sparsely** (see [`crate::machine_prep`]) instead
//! of densely. The numeric result is bit-faithful because absent entries are
//! NEG_INF, the identity of both semiring reduces.
//!
//! State layout (flat `Vec<f64>`, matching the JS Float64Arrays):
//!   - `core_m`: `K * S_td`  — M_k value (pre- or post-emission per phase)
//!   - `core_i`: `K * S_td`  — I_k value
//!   - `core_d`: `K * S_td`  — D_k (silent)
//!   - `flanking`: `N_FLANKING * S_td`
//!
//! Wasm-safe: the hot DP uses only `Vec<f64>` scratch and `f64` math.

// The DP deliberately uses explicit index loops that mirror the JS reference
// line-for-line (and whose visitation order is load-bearing for bit-identical
// logsumexp accumulation). Iterator rewrites would obscure that correspondence,
// so the `needless_range_loop` lint is intentionally allowed here.
#![allow(clippy::needless_range_loop)]

use crate::hmmer::{calc_match_occupancy, HmmerModel};
use crate::logmath::{safe_log, Semiring, NEG_INF};
use crate::machine_prep::{PreparedMachine, SparseEdge};

// Flanking state indices (identical to the JS constants).
const N: usize = 0;
const NX: usize = 1;
const B: usize = 2;
const E: usize = 3;
const CX: usize = 4;
const C: usize = 5;
const JX: usize = 6;
const J: usize = 7;
const N_FLANKING: usize = 8;

/// Options for [`build_fused_plan7`] (mirrors the JS `opts`).
#[derive(Clone, Copy, Debug)]
pub struct FusedOpts {
    pub multihit: bool,
    pub l: f64,
}

impl Default for FusedOpts {
    /// Matches the JS default `{ multihit: false, L: 400 }`.
    fn default() -> Self {
        FusedOpts {
            multihit: false,
            l: 400.0,
        }
    }
}

/// The fused-machine data (`fm`) built once and reused across sequences.
///
/// Mirrors the object returned by `buildFusedPlan7`, but the transducer is held
/// by reference to a [`PreparedMachine`] (sparse edges) rather than copied as a
/// dense tensor.
pub struct FusedPlan7<'a> {
    pub k: usize,
    pub n_aa: usize,
    pub s_td: usize,

    // Per-node core transitions (log-space).
    log_m_to_m: Vec<f64>,
    log_m_to_i: Vec<f64>,
    log_m_to_d: Vec<f64>,
    log_i_to_m: Vec<f64>,
    log_i_to_i: Vec<f64>,
    log_d_to_m: Vec<f64>,
    log_d_to_d: Vec<f64>,

    // Per-node emissions (log-space), flat K*n_aa.
    log_match_emit: Vec<f64>,
    log_ins_emit: Vec<f64>,

    // Local-mode begin entry, per node (log-space).
    log_b_entry: Vec<f64>,

    // Flanking weights (log-space scalars).
    log_n_loop: f64,
    log_n_to_b: f64,
    log_e_to_cx: f64,
    log_e_to_jx: f64,
    log_c_loop: f64,
    log_c_to_t: f64,
    log_j_loop: f64,
    log_j_to_b: f64,

    // Background/null emission (log-space), length n_aa.
    log_null_emit: Vec<f64>,

    // Amino-acid index -> transducer input token (0 = epsilon when unmapped).
    aa_to_td_in: Vec<usize>,

    // The prepared transducer (sparse edges + dense silent matrix).
    td: &'a PreparedMachine,
}

impl<'a> FusedPlan7<'a> {
    #[inline]
    fn silent(&self) -> &[f64] {
        &self.td.silent_dense
    }
    #[inline]
    fn emit_edges(&self, in_tok: usize, out_tok: usize) -> &[SparseEdge] {
        self.td.edges_for(in_tok, out_tok)
    }
}

/// Build the fused Plan7 data from a parsed HMMER model + prepared transducer.
///
/// Faithful port of `buildFusedPlan7`.
pub fn build_fused_plan7<'a>(
    model: &HmmerModel,
    transducer: &'a PreparedMachine,
    opts: FusedOpts,
) -> FusedPlan7<'a> {
    let k = model.nodes.len();
    let n_aa = model.alph.len();
    let s_td = transducer.n_states;

    // Per-node core transitions.
    let mut log_m_to_m = vec![0.0; k];
    let mut log_m_to_i = vec![0.0; k];
    let mut log_m_to_d = vec![0.0; k];
    let mut log_i_to_m = vec![0.0; k];
    let mut log_i_to_i = vec![0.0; k];
    let mut log_d_to_m = vec![0.0; k];
    let mut log_d_to_d = vec![0.0; k];
    for kk in 0..k {
        let n = &model.nodes[kk];
        log_m_to_m[kk] = safe_log(n.m_to_m);
        log_m_to_i[kk] = safe_log(n.m_to_i);
        log_m_to_d[kk] = safe_log(n.m_to_d);
        log_i_to_m[kk] = safe_log(n.i_to_m);
        log_i_to_i[kk] = safe_log(n.i_to_i);
        log_d_to_m[kk] = safe_log(n.d_to_m);
        log_d_to_d[kk] = safe_log(n.d_to_d);
    }

    // Per-node emissions, flat K*n_aa.
    let mut log_match_emit = vec![0.0; k * n_aa];
    let mut log_ins_emit = vec![0.0; k * n_aa];
    for kk in 0..k {
        for a in 0..n_aa {
            log_match_emit[kk * n_aa + a] = safe_log(model.nodes[kk].match_emit[a]);
            log_ins_emit[kk * n_aa + a] = safe_log(model.nodes[kk].ins_emit[a]);
        }
    }

    // Local-mode begin entry (occupancy-weighted).
    let occ = calc_match_occupancy(model);
    let mut z = 0.0_f64;
    for kk in 1..k {
        z += occ[kk] * (k - kk + 1) as f64;
    }
    let mut log_b_entry = vec![NEG_INF; k];
    if z > 0.0 {
        for kk in 0..k.saturating_sub(1) {
            // fused index kk maps to profile node kk+1
            log_b_entry[kk] = if occ[kk + 1] > 0.0 {
                (occ[kk + 1] / z).ln()
            } else {
                NEG_INF
            };
        }
        if k >= 1 {
            log_b_entry[k - 1] = NEG_INF; // last node: no entry in local
        }
    }

    // Flanking weights.
    let l = opts.l;
    let log_n_loop = (l / (l + 1.0)).ln();
    let log_n_to_b = (1.0 / (l + 1.0)).ln();
    let log_c_loop = (l / (l + 1.0)).ln();
    let log_c_to_t = (1.0 / (l + 1.0)).ln();

    let (log_e_to_cx, log_e_to_jx, log_j_loop, log_j_to_b) = if opts.multihit {
        (
            0.5_f64.ln(),
            0.5_f64.ln(),
            (l / (l + 1.0)).ln(),
            (1.0 / (l + 1.0)).ln(),
        )
    } else {
        (0.0, NEG_INF, NEG_INF, NEG_INF)
    };

    // Null model emissions.
    let mut log_null_emit = vec![0.0; n_aa];
    for a in 0..n_aa {
        log_null_emit[a] = safe_log(model.null_emit[a]);
    }

    // aa index -> transducer input token.
    let mut aa_to_td_in = vec![0usize; n_aa];
    for a in 0..n_aa {
        let sym = &model.alph[a];
        if let Some(pos) = transducer.input_alphabet.iter().position(|t| t == sym) {
            aa_to_td_in[a] = pos;
        }
        // else 0 (epsilon)
    }

    FusedPlan7 {
        k,
        n_aa,
        s_td,
        log_m_to_m,
        log_m_to_i,
        log_m_to_d,
        log_i_to_m,
        log_i_to_i,
        log_d_to_m,
        log_d_to_d,
        log_match_emit,
        log_ins_emit,
        log_b_entry,
        log_n_loop,
        log_n_to_b,
        log_e_to_cx,
        log_e_to_jx,
        log_c_loop,
        log_c_to_t,
        log_j_loop,
        log_j_to_b,
        log_null_emit,
        aa_to_td_in,
        td: transducer,
    }
}

// =========================================================================
// Internal helpers (sparse-aware where the JS used the dense tensor)
// =========================================================================

/// Transducer matvec over the DENSE silent matrix:
/// `result[dst] = reduce_src(v[src] + silent[src*S + dst])`.
/// Mirrors `tdMatvec(td_silent, ...)`.
fn td_matvec_silent(silent: &[f64], v: &[f64], s_td: usize, sem: Semiring, out: &mut [f64]) {
    for dst in 0..s_td {
        // Iterate src ascending — same order as the JS dense reduce.
        out[dst] = sem.reduce((0..s_td).map(|src| v[src] + silent[src * s_td + dst]));
    }
}

/// Propagate transducer silent transitions to a fixed point, in place on `v`.
/// Faithful port of `propagateTdSilent` (including the `v_td` base term, the
/// `1e-10` change threshold, and `maxIter = 100`).
fn propagate_td_silent(v: &mut [f64], silent: &[f64], s_td: usize, sem: Semiring) {
    let v_base: Vec<f64> = v.to_vec();
    let mut current: Vec<f64> = v.to_vec();
    let mut update = vec![NEG_INF; s_td];
    let mut next = vec![NEG_INF; s_td];
    for _ in 0..100 {
        td_matvec_silent(silent, &current, s_td, sem, &mut update);
        let mut changed = false;
        for i in 0..s_td {
            next[i] = sem.plus(v_base[i], update[i]);
            if (next[i] - current[i]).abs() > 1e-10 {
                changed = true;
            }
        }
        current.copy_from_slice(&next);
        if !changed {
            break;
        }
    }
    v.copy_from_slice(&current);
}

/// Plan7 state emits an amino acid; transducer consumes it and produces
/// `out_tok` (sparsely). Returns the new value vector over transducer states.
///
/// Sparse port of `tdEmitProduce`: for each aa with finite emit, fold its
/// `(aa_to_td_in[a], out_tok)` edges grouped by `dst` (src ascending) into a
/// per-dst column reduce, then `plus`-accumulate `aa_emit_log[a] + tdResult`.
fn td_emit_produce(
    fm: &FusedPlan7,
    v_td: &[f64],
    aa_emit_log: &[f64],
    out_tok: usize,
    sem: Semiring,
    result: &mut [f64],
) {
    let s_td = fm.s_td;
    for r in result.iter_mut().take(s_td) {
        *r = NEG_INF;
    }
    for a in 0..fm.n_aa {
        let emit = aa_emit_log[a];
        if emit == NEG_INF {
            continue;
        }
        let in_tok = fm.aa_to_td_in[a];
        let edges = fm.emit_edges(in_tok, out_tok);
        // Walk contiguous runs of equal dst (edges are sorted by (dst, src)).
        let mut i = 0usize;
        while i < edges.len() {
            let dst = edges[i].dst as usize;
            let mut j = i;
            while j < edges.len() && edges[j].dst as usize == dst {
                j += 1;
            }
            // Column reduce over src-ascending edges: reduce(v[src] + w).
            let td_result = sem.reduce(edges[i..j].iter().map(|e| v_td[e.src as usize] + e.log_weight));
            result[dst] = sem.plus(result[dst], emit + td_result);
            i = j;
        }
    }
}

/// Convenience: emit with `out_tok = 0` (transducer consumes, no output).
/// Mirrors `tdEmitSilent`.
#[inline]
fn td_emit_silent(
    fm: &FusedPlan7,
    v_td: &[f64],
    aa_emit_log: &[f64],
    sem: Semiring,
    result: &mut [f64],
) {
    td_emit_produce(fm, v_td, aa_emit_log, 0, sem, result);
}

/// Apply the transducer "delete" block `(in=0, out=out_tok)` to `src_vec`:
/// `out[dst] = reduce_src(src_vec[src] + td[0,out_tok,src,dst])` (sparse).
/// Returns NEG_INF for dst columns with no edge (reduce identity).
fn td_delete(fm: &FusedPlan7, src_vec: &[f64], out_tok: usize, sem: Semiring, out: &mut [f64]) {
    let s_td = fm.s_td;
    for o in out.iter_mut().take(s_td) {
        *o = NEG_INF;
    }
    let edges = fm.emit_edges(0, out_tok);
    let mut i = 0usize;
    while i < edges.len() {
        let dst = edges[i].dst as usize;
        let mut j = i;
        while j < edges.len() && edges[j].dst as usize == dst {
            j += 1;
        }
        out[dst] = sem.reduce(edges[i..j].iter().map(|e| src_vec[e.src as usize] + e.log_weight));
        i = j;
    }
}

/// Single-pass propagation of the silent flanking DAG + transducer-silent
/// closure at each flanking state. Faithful port of `propagateFlankingSilent`.
fn propagate_flanking_silent(flanking: &[f64], fm: &FusedPlan7, sem: Semiring) -> Vec<f64> {
    let s = fm.s_td;
    let mut result = flanking.to_vec();

    // E -> CX
    for st in 0..s {
        let v = result[E * s + st] + fm.log_e_to_cx;
        result[CX * s + st] = sem.plus(result[CX * s + st], v);
    }
    // E -> JX
    for st in 0..s {
        let v = result[E * s + st] + fm.log_e_to_jx;
        result[JX * s + st] = sem.plus(result[JX * s + st], v);
    }
    // NX -> B
    for st in 0..s {
        let v = result[NX * s + st] + fm.log_n_to_b;
        result[B * s + st] = sem.plus(result[B * s + st], v);
    }
    // JX -> B (after E -> JX)
    for st in 0..s {
        let v = result[JX * s + st] + fm.log_j_to_b;
        result[B * s + st] = sem.plus(result[B * s + st], v);
    }
    // NX -> N (n_loop)
    for st in 0..s {
        let v = result[NX * s + st] + fm.log_n_loop;
        result[N * s + st] = sem.plus(result[N * s + st], v);
    }
    // CX -> C (after E -> CX)
    for st in 0..s {
        let v = result[CX * s + st] + fm.log_c_loop;
        result[C * s + st] = sem.plus(result[C * s + st], v);
    }
    // JX -> J (after E -> JX)
    for st in 0..s {
        let v = result[JX * s + st] + fm.log_j_loop;
        result[J * s + st] = sem.plus(result[J * s + st], v);
    }

    // Transducer-silent closure at each flanking state.
    let silent = fm.silent();
    let mut tmp = vec![NEG_INF; s];
    for f in 0..N_FLANKING {
        tmp.copy_from_slice(&result[f * s..(f + 1) * s]);
        propagate_td_silent(&mut tmp, silent, s, sem);
        result[f * s..(f + 1) * s].copy_from_slice(&tmp);
    }
    result
}

// =========================================================================
// Main DP
// =========================================================================

/// Fused Plan7 + transducer Forward (or Viterbi) over `output_seq` (1-based
/// output token indices). Faithful port of `fusedPlan7Forward`.
pub fn fused_plan7_forward(fm: &FusedPlan7, output_seq: &[u32], sem: Semiring) -> f64 {
    let s = fm.s_td;
    let k = fm.k;
    let silent = fm.silent();

    let mut core_m = vec![NEG_INF; k * s];
    let mut core_i = vec![NEG_INF; k * s];
    let mut core_d = vec![NEG_INF; k * s];
    let mut flanking = vec![NEG_INF; N_FLANKING * s];

    // Scratch reused across init steps.
    let mut scratch = vec![NEG_INF; s];

    // --- Initialize ---
    // Step 1: S -> NX; propagate td_silent at NX.
    flanking[NX * s] = 0.0;
    {
        scratch.copy_from_slice(&flanking[NX * s..(NX + 1) * s]);
        propagate_td_silent(&mut scratch, silent, s, sem);
        flanking[NX * s..(NX + 1) * s].copy_from_slice(&scratch);
    }

    // Step 2: NX -> B, NX -> N.
    for st in 0..s {
        flanking[B * s + st] = flanking[NX * s + st] + fm.log_n_to_b;
        flanking[N * s + st] = flanking[NX * s + st] + fm.log_n_loop;
    }
    for &f in &[B, N] {
        scratch.copy_from_slice(&flanking[f * s..(f + 1) * s]);
        propagate_td_silent(&mut scratch, silent, s, sem);
        flanking[f * s..(f + 1) * s].copy_from_slice(&scratch);
    }

    // Step 3: B -> M_k (pre-emission entry).
    for kk in 0..k {
        for st in 0..s {
            core_m[kk * s + st] = fm.log_b_entry[kk] + flanking[B * s + st];
        }
    }

    // Step 4: M_k -> E (pre-emission local exit, weight 1).
    let mut e_val = vec![NEG_INF; s];
    let mut m_k = vec![NEG_INF; s];
    for kk in 0..k {
        m_k.copy_from_slice(&core_m[kk * s..(kk + 1) * s]);
        propagate_td_silent(&mut m_k, silent, s, sem);
        for st in 0..s {
            e_val[st] = sem.plus(e_val[st], m_k[st]);
        }
    }
    flanking[E * s..(E + 1) * s].copy_from_slice(&e_val);
    {
        scratch.copy_from_slice(&flanking[E * s..(E + 1) * s]);
        propagate_td_silent(&mut scratch, silent, s, sem);
        flanking[E * s..(E + 1) * s].copy_from_slice(&scratch);
    }

    // Step 5: E -> CX, E -> JX.
    for st in 0..s {
        flanking[CX * s + st] = flanking[E * s + st] + fm.log_e_to_cx;
        flanking[JX * s + st] = flanking[E * s + st] + fm.log_e_to_jx;
    }

    // Step 6: JX -> B (multi-hit increment).
    for st in 0..s {
        let jx_to_b = flanking[JX * s + st] + fm.log_j_to_b;
        flanking[B * s + st] = sem.plus(flanking[B * s + st], jx_to_b);
    }

    // Step 7: CX -> C, JX -> J.
    for st in 0..s {
        flanking[C * s + st] = flanking[CX * s + st] + fm.log_c_loop;
        flanking[J * s + st] = flanking[JX * s + st] + fm.log_j_loop;
    }
    for &f in &[CX, JX, C, J] {
        scratch.copy_from_slice(&flanking[f * s..(f + 1) * s]);
        propagate_td_silent(&mut scratch, silent, s, sem);
        flanking[f * s..(f + 1) * s].copy_from_slice(&scratch);
    }

    // Step 8: Multi-hit B -> M_k entries (from JX -> B increment only).
    {
        let mut jx_to_b = vec![NEG_INF; s];
        for st in 0..s {
            jx_to_b[st] = flanking[JX * s + st] + fm.log_j_to_b;
        }
        propagate_td_silent(&mut jx_to_b, silent, s, sem);
        for kk in 0..k {
            for st in 0..s {
                core_m[kk * s + st] =
                    sem.plus(core_m[kk * s + st], fm.log_b_entry[kk] + jx_to_b[st]);
            }
        }
    }

    // Initial silent Plan7 emissions.
    emit_silent_core(&mut core_m, &mut core_i, fm, sem);

    let lo = output_seq.len();
    if lo == 0 {
        return get_terminal_val(&flanking, fm, sem);
    }

    for p in 0..lo {
        let out_tok = output_seq[p] as usize;

        emit_output_step(
            &mut core_m,
            &mut core_i,
            &mut core_d,
            &mut flanking,
            out_tok,
            fm,
            sem,
        );

        let routed = route_post_emission(&core_m, &core_i, &core_d, &flanking, fm, sem);
        core_m = routed.0;
        core_i = routed.1;
        core_d = routed.2;
        flanking = routed.3;

        emit_silent_core(&mut core_m, &mut core_i, fm, sem);
    }

    get_terminal_val(&flanking, fm, sem)
}

/// Viterbi = Forward with the max-plus semiring. Mirrors `fusedPlan7Viterbi`.
#[inline]
pub fn fused_plan7_viterbi(fm: &FusedPlan7, output_seq: &[u32]) -> f64 {
    fused_plan7_forward(fm, output_seq, Semiring::MaxPlus)
}

// =========================================================================
// Internal DP steps
// =========================================================================

/// Plan7 core emits, transducer consumes silently. Modifies `core_m`/`core_i`
/// in place. Faithful port of `_emitSilentCore`.
fn emit_silent_core(core_m: &mut [f64], core_i: &mut [f64], fm: &FusedPlan7, sem: Semiring) {
    let s = fm.s_td;
    let n_aa = fm.n_aa;
    let mut result = vec![NEG_INF; s];
    let mut v = vec![NEG_INF; s];

    for kk in 0..fm.k {
        // M_k
        v.copy_from_slice(&core_m[kk * s..(kk + 1) * s]);
        td_emit_silent(
            fm,
            &v,
            &fm.log_match_emit[kk * n_aa..(kk + 1) * n_aa],
            sem,
            &mut result,
        );
        for st in 0..s {
            core_m[kk * s + st] = sem.plus(core_m[kk * s + st], result[st]);
        }
        // I_k
        v.copy_from_slice(&core_i[kk * s..(kk + 1) * s]);
        td_emit_silent(
            fm,
            &v,
            &fm.log_ins_emit[kk * n_aa..(kk + 1) * n_aa],
            sem,
            &mut result,
        );
        for st in 0..s {
            core_i[kk * s + st] = sem.plus(core_i[kk * s + st], result[st]);
        }
    }
}

/// Process one output token: Plan7 emits -> transducer produces output.
/// In-place: overwrites the four state arrays with the post-emission values.
/// Faithful port of `_emitOutputStep`.
fn emit_output_step(
    core_m: &mut Vec<f64>,
    core_i: &mut Vec<f64>,
    core_d: &mut Vec<f64>,
    flanking: &mut Vec<f64>,
    out_tok: usize,
    fm: &FusedPlan7,
    sem: Semiring,
) {
    let s = fm.s_td;
    let k = fm.k;
    let n_aa = fm.n_aa;

    let mut new_core_m = vec![NEG_INF; k * s];
    let mut new_core_i = vec![NEG_INF; k * s];
    let mut new_core_d = vec![NEG_INF; k * s];
    let mut new_flanking = vec![NEG_INF; N_FLANKING * s];

    let mut emitted = vec![NEG_INF; s];
    let mut v = vec![NEG_INF; s];

    // 1. Core M_k emits, td produces output.
    for kk in 0..k {
        v.copy_from_slice(&core_m[kk * s..(kk + 1) * s]);
        td_emit_produce(
            fm,
            &v,
            &fm.log_match_emit[kk * n_aa..(kk + 1) * n_aa],
            out_tok,
            sem,
            &mut emitted,
        );
        new_core_m[kk * s..(kk + 1) * s].copy_from_slice(&emitted);
    }

    // 2. Core I_k emits, td produces output.
    for kk in 0..k {
        v.copy_from_slice(&core_i[kk * s..(kk + 1) * s]);
        td_emit_produce(
            fm,
            &v,
            &fm.log_ins_emit[kk * n_aa..(kk + 1) * n_aa],
            out_tok,
            sem,
            &mut emitted,
        );
        new_core_i[kk * s..(kk + 1) * s].copy_from_slice(&emitted);
    }

    // 3. Flanking N/C/J emit background, td produces output.
    let mut n_emitted = vec![NEG_INF; s];
    let mut c_emitted = vec![NEG_INF; s];
    let mut j_emitted = vec![NEG_INF; s];
    v.copy_from_slice(&flanking[N * s..(N + 1) * s]);
    td_emit_produce(fm, &v, &fm.log_null_emit, out_tok, sem, &mut n_emitted);
    v.copy_from_slice(&flanking[C * s..(C + 1) * s]);
    td_emit_produce(fm, &v, &fm.log_null_emit, out_tok, sem, &mut c_emitted);
    v.copy_from_slice(&flanking[J * s..(J + 1) * s]);
    td_emit_produce(fm, &v, &fm.log_null_emit, out_tok, sem, &mut j_emitted);

    // 4. Transducer "delete" (in=0, out=out_tok) applied to all states.
    let mut td_del = vec![NEG_INF; s];
    for kk in 0..k {
        let base = kk * s;
        td_delete(fm, &core_m[base..base + s], out_tok, sem, &mut td_del);
        for dst in 0..s {
            new_core_m[base + dst] = sem.plus(new_core_m[base + dst], td_del[dst]);
        }
        td_delete(fm, &core_i[base..base + s], out_tok, sem, &mut td_del);
        for dst in 0..s {
            new_core_i[base + dst] = sem.plus(new_core_i[base + dst], td_del[dst]);
        }
        td_delete(fm, &core_d[base..base + s], out_tok, sem, &mut td_del);
        new_core_d[base..base + s].copy_from_slice(&td_del[..s]);
    }

    for f in 0..N_FLANKING {
        let base = f * s;
        td_delete(fm, &flanking[base..base + s], out_tok, sem, &mut td_del);
        for dst in 0..s {
            new_flanking[base + dst] = if f == N {
                sem.plus(n_emitted[dst], td_del[dst])
            } else if f == C {
                sem.plus(c_emitted[dst], td_del[dst])
            } else if f == J {
                sem.plus(j_emitted[dst], td_del[dst])
            } else {
                td_del[dst]
            };
        }
    }

    *core_m = new_core_m;
    *core_i = new_core_i;
    *core_d = new_core_d;
    *flanking = new_flanking;
}

/// Route after emissions: inner scan over core positions for Mx->next / D->next
/// chains, insert self-loops, flanking routing, and the B-entry -> E -> CX/JX
/// closure. Faithful port of `_routePostEmission`. Returns the four new arrays.
fn route_post_emission(
    core_m: &[f64],
    core_i: &[f64],
    core_d: &[f64],
    flanking: &[f64],
    fm: &FusedPlan7,
    sem: Semiring,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let s = fm.s_td;
    let k = fm.k;
    let silent = fm.silent();

    let mut m_incoming = vec![NEG_INF; s];
    let mut d_incoming = vec![NEG_INF; s];
    let mut e_accum = vec![NEG_INF; s];

    let mut m_arriving = vec![NEG_INF; k * s];
    let mut new_core_i = vec![NEG_INF; k * s];

    // Reused scratch.
    let mut d_at_k = vec![NEG_INF; s];
    let mut m_pre_k = vec![NEG_INF; s];
    let mut mx_k = vec![NEG_INF; s];
    let mut ix_k = vec![NEG_INF; s];

    for kk in 0..k {
        // D_k: combine emit-step D_k with incoming from left, then close silent.
        for st in 0..s {
            d_at_k[st] = sem.plus(core_d[kk * s + st], d_incoming[st]);
        }
        propagate_td_silent(&mut d_at_k, silent, s, sem);

        // Pre-emission M_k from routing (ONLY m_incoming).
        m_pre_k.copy_from_slice(&m_incoming);
        propagate_td_silent(&mut m_pre_k, silent, s, sem);

        // Post-emission Mx_k.
        mx_k.copy_from_slice(&core_m[kk * s..(kk + 1) * s]);
        propagate_td_silent(&mut mx_k, silent, s, sem);

        // Post-emission Ix_k.
        ix_k.copy_from_slice(&core_i[kk * s..(kk + 1) * s]);
        propagate_td_silent(&mut ix_k, silent, s, sem);

        // E contributions: ONLY pre-emission M_k and D_k.
        for st in 0..s {
            e_accum[st] = sem.plus(e_accum[st], m_pre_k[st]);
            e_accum[st] = sem.plus(e_accum[st], d_at_k[st]);
        }

        // Persist pre-emission M_k.
        m_arriving[kk * s..(kk + 1) * s].copy_from_slice(&m_pre_k);

        // Routing from Mx_k / Ix_k / D_k.
        let mut new_m_incoming = vec![NEG_INF; s];
        let mut new_d_incoming = vec![NEG_INF; s];
        for st in 0..s {
            let m_to_next = mx_k[st] + fm.log_m_to_m[kk];
            let i_from_m = mx_k[st] + fm.log_m_to_i[kk];
            let d_from_m = mx_k[st] + fm.log_m_to_d[kk];

            let m_from_i = ix_k[st] + fm.log_i_to_m[kk];
            let i_self = ix_k[st] + fm.log_i_to_i[kk];

            let m_from_d = d_at_k[st] + fm.log_d_to_m[kk];
            let d_from_d = d_at_k[st] + fm.log_d_to_d[kk];

            new_m_incoming[st] = sem.plus(m_to_next, sem.plus(m_from_i, m_from_d));
            new_d_incoming[st] = sem.plus(d_from_m, d_from_d);

            new_core_i[kk * s + st] = sem.plus(i_from_m, i_self);
        }

        m_incoming = new_m_incoming;
        d_incoming = new_d_incoming;
    }

    // Build new flanking.
    let mut new_flanking = vec![NEG_INF; N_FLANKING * s];
    for st in 0..s {
        new_flanking[NX * s + st] = flanking[N * s + st];
        new_flanking[CX * s + st] = flanking[C * s + st];
        new_flanking[JX * s + st] = flanking[J * s + st];
        new_flanking[E * s + st] = e_accum[st];
    }

    // Propagate silent flanking.
    let closed_flanking = propagate_flanking_silent(&new_flanking, fm, sem);

    // Final core_m = pre-emission from routing + new B entry.
    let mut new_core_m = vec![NEG_INF; k * s];
    let new_core_d = vec![NEG_INF; k * s];
    let mut result_flanking = closed_flanking.clone();

    // b_val_closed = closed_flanking[B].
    for kk in 0..k {
        for st in 0..s {
            new_core_m[kk * s + st] = sem.plus(
                m_arriving[kk * s + st],
                fm.log_b_entry[kk] + closed_flanking[B * s + st],
            );
        }
    }

    // New B -> M_k -> E closure within this step.
    let mut e_from_b = vec![NEG_INF; s];
    let mut b_mk = vec![NEG_INF; s];
    for kk in 0..k {
        for st in 0..s {
            b_mk[st] = fm.log_b_entry[kk] + closed_flanking[B * s + st];
        }
        propagate_td_silent(&mut b_mk, silent, s, sem);
        for st in 0..s {
            e_from_b[st] = sem.plus(e_from_b[st], b_mk[st]);
        }
    }
    // Add B-entry E contribution.
    for st in 0..s {
        result_flanking[E * s + st] = sem.plus(result_flanking[E * s + st], e_from_b[st]);
    }
    // E -> CX (then CX -> C).
    let mut e_closed = e_from_b.clone();
    propagate_td_silent(&mut e_closed, silent, s, sem);
    for st in 0..s {
        let cx_inc = e_closed[st] + fm.log_e_to_cx;
        result_flanking[CX * s + st] = sem.plus(result_flanking[CX * s + st], cx_inc);
        let c_inc = cx_inc + fm.log_c_loop;
        result_flanking[C * s + st] = sem.plus(result_flanking[C * s + st], c_inc);
    }
    // E -> JX -> J and JX -> B (multi-hit), plus extra B -> M_k entries.
    for st in 0..s {
        let jx_inc = e_closed[st] + fm.log_e_to_jx;
        result_flanking[JX * s + st] = sem.plus(result_flanking[JX * s + st], jx_inc);
        let j_inc = jx_inc + fm.log_j_loop;
        result_flanking[J * s + st] = sem.plus(result_flanking[J * s + st], j_inc);
        let b_inc = jx_inc + fm.log_j_to_b;
        result_flanking[B * s + st] = sem.plus(result_flanking[B * s + st], b_inc);
        for kk in 0..k {
            new_core_m[kk * s + st] =
                sem.plus(new_core_m[kk * s + st], fm.log_b_entry[kk] + b_inc);
        }
    }

    (new_core_m, new_core_i, new_core_d, result_flanking)
}

/// Terminal value: `Cx -> T`, then close silent, read state `S_td - 1`.
/// Faithful port of `_getTerminalVal`.
fn get_terminal_val(flanking: &[f64], fm: &FusedPlan7, sem: Semiring) -> f64 {
    let s = fm.s_td;
    let mut cx_val = vec![NEG_INF; s];
    for st in 0..s {
        cx_val[st] = flanking[CX * s + st] + fm.log_c_to_t;
    }
    propagate_td_silent(&mut cx_val, fm.silent(), s, sem);
    cx_val[s - 1]
}
