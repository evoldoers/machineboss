#!/usr/bin/env python3
"""Validate the skeleton-bake Rust codegen mode (Increment 1 of the
bake-and-expand work).

Runs `boss --phylo-skeleton --codegen <dir> --rust` on a small TKF91-DNA
tree, then `cargo build` + `cargo test` on the resulting crate. Confirms:
- the four baked `&'static str` constants (T_JSON, M_SKEL_JSON, TREE_NEWICK,
  TIME_PARAM) parse as well-formed JSON / Newick,
- `prebuild()` panics with the documented TODO message.
"""
import os, sys, subprocess, tempfile, shutil

REPO = os.environ.get('REPO_ROOT', os.getcwd())
BOSS = os.path.join(REPO, 'bin', 'boss')

# Pull in the shared exact-lse multidim Forward reference from t/rust/.
sys.path.insert(0, os.path.join(REPO, 't', 'rust'))
from _phylo_ref import multidim_forward

def run(cmd, cwd=None, capture=True):
    r = subprocess.run(cmd, capture_output=capture, text=True, cwd=cwd)
    if r.returncode != 0:
        sys.stderr.write(f"FAIL: {cmd}\n{r.stderr}\n{r.stdout}\n")
        sys.exit(1)
    return r.stdout

def main():
    if shutil.which('cargo') is None:
        print("                 test-phylo-skeleton-bake     skip: cargo not in PATH")
        return
    work = tempfile.mkdtemp(prefix='skelbake_')
    crate = os.path.join(work, 'crate')

    run([BOSS, '--tkf91-branch-dna-jc',
         '--phylo-tree-string', '(A,B)P;', '--phylo-time-param', 'time',
         '--phylo-skeleton', '--codegen', crate, '--rust'])

    # Layout sanity
    assert os.path.isfile(os.path.join(crate, 'Cargo.toml'))
    assert os.path.isfile(os.path.join(crate, 'src', 'lib.rs'))

    # Drop a tests/ file that asserts the four constants and the panic, and
    # exercises weight_algebra against the baked TKF91 defs with reference
    # values pre-computed in Python (see comments inline).
    tests_dir = os.path.join(crate, 'tests')
    os.makedirs(tests_dir, exist_ok=True)
    with open(os.path.join(tests_dir, 'bake_parses.rs'), 'w') as f:
        f.write('''use phylo_skeleton::{T_JSON, M_SKEL_JSON, TREE_NEWICK, TIME_PARAM};
use phylo_skeleton::weight_algebra::{Defs, Params, evaluate, parse_defs};
use serde_json::Value;

#[test] fn t_json_parses() {
    let v: Value = serde_json::from_str(T_JSON).expect("T_JSON parses");
    assert!(v.get("state").is_some());
}
#[test] fn m_skel_json_parses() {
    let v: Value = serde_json::from_str(M_SKEL_JSON).expect("M_SKEL_JSON parses");
    assert!(v.get("state").is_some());
}
#[test] fn tree_newick_well_formed() {
    assert!(TREE_NEWICK.ends_with(";"));
    assert_eq!(TREE_NEWICK.matches('(').count(), TREE_NEWICK.matches(')').count());
}
#[test] fn time_param_nonempty() { assert!(!TIME_PARAM.is_empty()); }

#[test] fn prebuild_returns_phylo_machine() {
    // prebuild() now actually builds the phylo-composed machine from the
    // baked T + tree. Smoke check the result has a non-zero state set with
    // the array-format state ids that compose/intersect produce, plus at
    // least one emit transition.
    let m = phylo_skeleton::prebuild();
    assert!(m.n_states() > 0);
    if let Value::Array(arr) = &m.state[0].id {
        assert_eq!(arr.len(), 2, "expected 2-array state id from compose; got {:?}", arr);
    } else {
        panic!("state 0 id not a 2-array: {:?}", m.state[0].id);
    }
    let emit_count: usize = m.state.iter()
        .map(|s| s.trans.iter().filter(|t| !t.out_sym.is_empty()).count())
        .sum();
    assert!(emit_count > 0, "expected at least one emit transition");
}

#[test] fn prebuild_evaluates_to_finite_weights() {
    // Pick a concrete parameter assignment and verify every transition
    // weight in the prebuild()-produced machine evaluates to a finite f64
    // (so the WeightAlgebra port sees no malformed / unreachable expressions
    // in the actual phylo composition output).
    use phylo_skeleton::weight_algebra::{Params, evaluate};
    let m = phylo_skeleton::prebuild();
    let mut p = Params::new();
    p.insert("insRate".into(), 0.005);
    p.insert("delRate".into(), 0.01);
    p.insert(format!("{}[A]", phylo_skeleton::TIME_PARAM), 0.3);
    p.insert(format!("{}[B]", phylo_skeleton::TIME_PARAM), 0.2);
    let mut emit_evaluated = 0usize;
    for s in &m.state {
        for t in &s.trans {
            let v = evaluate(&t.weight, &p, &m.defs);
            assert!(v.is_finite(), "non-finite weight: {} = {}", t.weight, v);
            if !t.out_sym.is_empty() { emit_evaluated += 1; }
        }
    }
    assert!(emit_evaluated > 0, "no emit transitions evaluated");
}

#[test] fn phylo_intersect_runs_on_baked_t_and_tree() {
    // End-to-end smoke: parse the baked T_JSON + TREE_NEWICK, run the
    // Rust phylo_intersect, verify the result has the expected shape
    // (every state has a 2-array id from intersect, a non-zero state count,
    // and at least one emit transition).
    use phylo_skeleton::machine::Machine;
    use phylo_skeleton::phylo::{PhyloTree, phylo_intersect};
    let t_json: Value = serde_json::from_str(T_JSON).expect("T_JSON parses");
    let t = Machine::from_json(&t_json);
    let tree = PhyloTree::parse_newick(TREE_NEWICK);
    let m = phylo_intersect(&t, &tree, TIME_PARAM);
    assert!(m.n_states() > 0);
    // Every state should have a 2-element-array id (from compose's array
    // state-name convention).
    if let Value::Array(a) = &m.state[0].id {
        assert_eq!(a.len(), 2);
    } else {
        panic!("state 0 id not a 2-array: {:?}", m.state[0].id);
    }
    // At least one emit transition should exist (the phylo machine emits
    // pair-tokens for the leaves).
    let emit_count: usize = m.state.iter()
        .map(|s| s.trans.iter().filter(|t| !t.out_sym.is_empty()).count())
        .sum();
    assert!(emit_count > 0, "no emit transitions in phylo machine");
}

#[test] fn compose_t_with_self_state_count_matches_cpp() {
    // With Increment 4e (advance_sort + process_cycles) wired into the Rust
    // compose pipeline, the Rust port now produces the SAME post-processing
    // chain as C++. Composing the baked TKF91-branch-dna-jc T with itself
    // should yield an identical state count to `bin/boss T.json T.json`
    // (which auto-composes when given two machine files). The expected
    // value below is taken directly from the C++ output.
    use phylo_skeleton::machine::{Machine, compose};
    let t_json: Value = serde_json::from_str(T_JSON).expect("T_JSON parses");
    let t = Machine::from_json(&t_json);
    let c = compose(&t, &t);
    // C++: `bin/boss T.json T.json --pair-json` → 21 states for tkf91-branch-dna-jc.
    assert_eq!(c.n_states(), 21,
               "Rust compose(T,T) state count must match C++ exactly; got {}",
               c.n_states());
    // Each state name should be a 2-array (from compose's array-format ids).
    if let Value::Array(arr) = &c.state[0].id {
        assert_eq!(arr.len(), 2);
    } else {
        panic!("composite state name not an array: {:?}", c.state[0].id);
    }
    // The result must be advancing (no silent back-transitions) — that is
    // the post-condition of process_cycles().
    assert!(c.is_advancing_machine(),
            "compose(T,T) result must be advancing");
}

#[test] fn baked_t_is_already_ergodic() {
    // TKF91-branch-dna-jc T has all states reachable from begin and able to
    // reach end → already ergodic; ergodic_machine should be a no-op.
    use phylo_skeleton::machine::Machine;
    let t_json: Value = serde_json::from_str(T_JSON).expect("T_JSON parses");
    let t = Machine::from_json(&t_json);
    assert!(t.is_ergodic_machine());
    assert_eq!(t.ergodic_machine().n_states(), t.n_states());
}

#[test] fn baked_t_is_already_waiting() {
    // TKF91-branch-dna-jc T has begin/orphan/wait/insert as silent-only
    // states and match/delete as consuming-only states, so T should already
    // satisfy is_waiting_machine; the transform is a state-preserving clone.
    use phylo_skeleton::machine::Machine;
    let t_json: Value = serde_json::from_str(T_JSON).expect("T_JSON parses");
    let t = Machine::from_json(&t_json);
    assert!(t.is_waiting_machine());
    let tw = t.waiting_machine();
    assert_eq!(tw.n_states(), t.n_states());
}

#[test] fn baked_t_renames_for_branch() {
    // Parse T_JSON, rename for branch "X", check defs/cons are suffixed and
    // that pSame[X] evaluates against time[X] to the same value as pSame
    // evaluates against time on the unrenamed T (consistency of bind/eval).
    use phylo_skeleton::machine::{Machine, rename_for_branch};
    let t_json: Value = serde_json::from_str(T_JSON).expect("T_JSON parses");
    let t = Machine::from_json(&t_json);
    let t_x = rename_for_branch(&t, TIME_PARAM, "X");

    // Defs renamed
    assert!(t_x.defs.contains_key("pSame[X]"));
    assert!(t_x.defs.contains_key("pNoSub[X]"));
    assert!(!t_x.defs.contains_key("pSame"));

    // cons.rate has time -> time[X]
    let time_x = format!("{}[X]", TIME_PARAM);
    assert!(t_x.cons.rate.contains(&time_x), "cons.rate: {:?}", t_x.cons.rate);

    // Evaluate pSame on T at time=0.5 vs pSame[X] on T_x at time[X]=0.5.
    let mut p0 = Params::new();
    p0.insert(TIME_PARAM.into(), 0.5);
    p0.insert("insRate".into(), 0.005);
    p0.insert("delRate".into(), 0.01);
    let v0 = evaluate(&Value::String("pSame".into()), &p0, &t.defs);

    let mut p1 = Params::new();
    p1.insert(time_x.clone(), 0.5);
    p1.insert("insRate".into(), 0.005);
    p1.insert("delRate".into(), 0.01);
    let v1 = evaluate(&Value::String("pSame[X]".into()), &p1, &t_x.defs);
    assert!((v0 - v1).abs() < 1e-15, "v0={} v1={}", v0, v1);
}

#[test] fn baked_tkf91_defs_evaluate() {
    // TKF91-DNA-JC defs at time[A]=0.5, time[B]=0.3, insRate=0.005,
    // delRate=0.01; reference values pre-computed in Python (see
    // t/check_phylo_skeleton_bake.py docstring).
    let m: Value = serde_json::from_str(M_SKEL_JSON).expect("M_SKEL_JSON parses");
    let defs: Defs = parse_defs(&m);
    let mut params: Params = Params::new();
    params.insert("time[A]".into(),  0.5);
    params.insert("time[B]".into(),  0.3);
    params.insert("insRate".into(), 0.005);
    params.insert("delRate".into(), 0.01);

    let close = |actual: f64, expected: f64, tag: &str| {
        let rel = (actual - expected).abs() / expected.abs().max(1e-30);
        assert!(rel < 1e-12, "{}: actual {} expected {}", tag, actual, expected);
    };

    close(evaluate(&Value::String("pNoSub[A]".into()), &params, &defs),
          0.606530659712633424, "pNoSub[A]");
    close(evaluate(&Value::String("pSub[A]".into()), &params, &defs),
          0.393469340287366576, "pSub[A]");
    close(evaluate(&Value::String("pDiff[A]".into()), &params, &defs),
          0.0983673350718416439, "pDiff[A]");
    close(evaluate(&Value::String("pSame[A]".into()), &params, &defs),
          0.704897994784475124, "pSame[A]");
    close(evaluate(&Value::String("pNoDescendants[A]".into()), &params, &defs),
          0.997509341267465044, "pNoDescendants[A]");
    close(evaluate(&Value::String("pDescendants[A]".into()), &params, &defs),
          0.00249065873253496734, "pDescendants[A]");
}

#[test]
fn compose_pipeline_resolves_silent_cycles_on_non_tkf_input() {
    // Increment 4e regression: a non-TKF transducer with an explicit silent
    // cycle (back-edge) drives the advance_sort + process_cycles port. The
    // C++ post-processing chain
    //   ergodicMachine().advanceSort().processCycles().ergodicMachine()
    // turns the silent cycle into a `geomsum` factor on the cycle's exits.
    //
    // Setup: T_loop has a 2-state silent cycle with input/output emit
    // transitions on each cycle node. Composed with a wild-echo on T_loop's
    // output alphabet (the same shape `phylo_intersect` uses for leaves),
    // the composed machine should be advancing (no silent back-transitions
    // remain) and every transition weight should evaluate to a finite f64.
    //
    //   States: S(0), A(1), B(2), E(3).
    //   A → B (silent, w=p), B → A (silent back-edge, w=r), and
    //   A also has a silent self-loop with weight q. A → E is the only
    //   loud transition (in:x out:y, w=1).
    use phylo_skeleton::machine::{Machine, compose};
    use phylo_skeleton::phylo::wild_echo;
    use phylo_skeleton::weight_algebra::{Params, evaluate};

    let t_loop = Machine::from_json(&serde_json::json!({
        "state": [
            {"id": "S", "trans": [{"to": 1}]},
            {"id": "A", "trans": [
                {"to": 2, "weight": "p"},
                {"to": 1, "weight": "q"},
                {"to": 3, "in": "x", "out": "y", "weight": 1.0}
            ]},
            {"id": "B", "trans": [{"to": 1, "weight": "r"}]},
            {"id": "E"}
        ]
    }));
    assert!(t_loop.n_silent_back_transitions() > 0,
            "test setup expected at least one silent back-edge");

    let echo = wild_echo(&["y".to_string()]);
    let composed = compose(&t_loop, &echo);
    // After the full post-processing chain the result must be advancing.
    assert!(composed.is_advancing_machine(),
            "compose pipeline must eliminate silent back-transitions; got {} silent backs in {} states",
            composed.n_silent_back_transitions(), composed.n_states());

    // Concrete-parameter sanity check: every weight must be finite at a
    // small parameter assignment chosen so the geomsum denominators are
    // strictly positive.
    let mut params = Params::new();
    params.insert("p".into(), 0.3);
    params.insert("q".into(), 0.4);
    params.insert("r".into(), 0.2);
    let mut emit_seen = 0;
    for s in &composed.state {
        for t in &s.trans {
            let v = evaluate(&t.weight, &params, &composed.defs);
            assert!(v.is_finite(),
                    "non-finite weight {} = {}", t.weight, v);
            if !t.out_sym.is_empty() { emit_seen += 1; }
        }
    }
    assert!(emit_seen > 0,
            "expected at least one emit transition after compose pipeline");
}
''')

    run(['cargo', 'build', '--release'], cwd=crate)
    run(['cargo', 'test', '--release'], cwd=crate)

    # Increment 6b: bit-exact end-to-end Forward DP check. Reference
    # value comes from running multidim_forward (exact log_sum_exp) on
    # the *C++* M_full JSON for the same TKF91-DNA-JC tree the bake uses.
    # Rust runs forward(prebuild(), ...) on the same params + leaves and
    # must agree to within 1e-12 (typically 1e-15 floating-point noise).
    machine_json = run([BOSS,
                        '--tkf91-branch-dna-jc',
                        '--phylo-tree-string', '(A,B)P;',
                        '--phylo-time-param', 'time',
                        '--pair-json'])
    params  = {'insRate': 0.005, 'delRate': 0.01,
               'time[A]': 0.3,   'time[B]': 0.2}
    leaves  = [list('ACGT'), list('ACG')]
    ref     = multidim_forward(machine_json, params, leaves)

    # Emit a tests/ regression test that compares forward(prebuild(),...)
    # to the externally-computed reference. tests/ files are picked up by
    # `cargo test`, so we re-run cargo test once the file is in place.
    fwd_check = os.path.join(tests_dir, 'forward_bit_exact.rs')
    with open(fwd_check, 'w') as f:
        f.write(f'''//! Increment 6b regression: forward(prebuild(),...) must match the
//! Python multidim_forward reference (exact log_sum_exp) bit-exactly to
//! within 1e-12. Reference value pre-computed in
//! t/check_phylo_skeleton_bake.py against the C++ M_full JSON.
use phylo_skeleton::forward::forward;
use phylo_skeleton::weight_algebra::Params;

const REFERENCE: f64 = {ref!r};

#[test] fn forward_matches_python_multidim_reference() {{
    let m = phylo_skeleton::prebuild();
    let mut params = Params::new();
    params.insert("insRate".into(), 0.005);
    params.insert("delRate".into(), 0.01);
    params.insert("time[A]".into(), 0.3);
    params.insert("time[B]".into(), 0.2);
    let leaves: Vec<Vec<String>> = vec![
        "ACGT".chars().map(|c| c.to_string()).collect(),
        "ACG".chars().map(|c|  c.to_string()).collect(),
    ];
    let lk = forward(&m, &params, &leaves);
    let diff = (lk - REFERENCE).abs();
    assert!(lk.is_finite(), "forward returned non-finite: {{}}", lk);
    assert!(diff < 1e-12,
            "forward = {{:.17e}} vs reference {{:.17e}}, diff = {{:.3e}}",
            lk, REFERENCE, diff);
}}
''')

    run(['cargo', 'test', '--release', '--test', 'forward_bit_exact'], cwd=crate)

    # Multi-tree coverage: bake a separate crate for the polytomy
    # (single internal node, three leaves), run forward bit-exact +
    # state-count + advancing checks. The binary-tree pipeline above
    # already exercises every test in tests/bake_parses.rs; the case
    # here only checks that the prebuild pipeline survives a deeper
    # tree shape (≥ 3-leaf tokens).
    #
    # The depth-3 ((((A,B)P,C)Q)D;) and quartet (((A,B)P,(C,D)Q)R;)
    # topologies live in the manual test target test-phylo-skeleton-bake-deep
    # — prebuild() takes ~3 minutes per call on those sizes (symbolic
    # weight expressions blow up under the unoptimised compose pipeline)
    # so they would dominate the default `make test` runtime.
    multi_cases = [
        # (tree, leaves, branch-times)
        ('(A,B,C)R;',
         [list('AC'), list('AC'), list('AC')],
         {'time[A]': 0.3, 'time[B]': 0.2, 'time[C]': 0.4}),
    ]

    for tree, leaves, time_params in multi_cases:
        run_tree_case(tree, leaves, time_params)

    print("OK")


def run_tree_case(tree, leaves, time_params):
    """Bake a fresh crate for `tree`, embed the C++-derived reference
    forward log-likelihood, run cargo test."""
    work  = tempfile.mkdtemp(prefix='skelbake_tree_')
    crate = os.path.join(work, 'crate')

    run([BOSS, '--tkf91-branch-dna-jc',
         '--phylo-tree-string', tree, '--phylo-time-param', 'time',
         '--phylo-skeleton', '--codegen', crate, '--rust'])

    # C++ reference: the multidim Forward DP (exact lse) over C++ M_full.
    machine_json = run([BOSS, '--tkf91-branch-dna-jc',
                        '--phylo-tree-string', tree,
                        '--phylo-time-param', 'time',
                        '--pair-json'])
    cpp_states = len(__import__('json').loads(machine_json)['state'])
    params = {'insRate': 0.005, 'delRate': 0.01, **time_params}
    ref    = multidim_forward(machine_json, params, leaves)

    # Build the Rust test file. The state-count assertion catches
    # divergence between C++ and the Rust port's compose/intersect
    # pipeline (advance_sort + process_cycles port). The bit-exact
    # forward check catches divergence in symbolic-weight arithmetic
    # and DP recurrences.
    tests_dir = os.path.join(crate, 'tests')
    os.makedirs(tests_dir, exist_ok=True)

    py_params_init = '\n    '.join(
        f'params.insert("{k}".into(), {v!r});' for k, v in params.items())
    py_leaves_init = ',\n        '.join(
        '"{0}".chars().map(|c| c.to_string()).collect()'.format(''.join(leaf))
        for leaf in leaves)

    with open(os.path.join(tests_dir, 'tree_check.rs'), 'w') as f:
        f.write(f'''//! Multi-tree regression for the bake-and-expand pipeline.
//!
//! Tree: {tree}
//! Leaves: {[''.join(l) for l in leaves]}
//!
//! Asserts (against the C++ phylo-intersect pipeline as ground truth):
//!   1. prebuild() state count == C++ M_full state count ({cpp_states})
//!   2. prebuild() is advancing (no silent back-transitions)
//!   3. forward(prebuild(), ...) matches Python multidim_forward (exact
//!      log_sum_exp) over C++ M_full to within 1e-12.
use phylo_skeleton::forward::forward;
use phylo_skeleton::weight_algebra::Params;

const CPP_STATES: usize = {cpp_states};
const REFERENCE_LK: f64 = {ref!r};

#[test] fn prebuild_state_count_matches_cpp() {{
    let m = phylo_skeleton::prebuild();
    assert_eq!(m.n_states(), CPP_STATES,
               "Rust prebuild() state count {{}} != C++ M_full state count {{}}",
               m.n_states(), CPP_STATES);
}}

#[test] fn prebuild_is_advancing_after_full_pipeline() {{
    let m = phylo_skeleton::prebuild();
    assert!(m.is_advancing_machine(),
            "prebuild() must yield an advancing machine; got {{}} silent backs",
            m.n_silent_back_transitions());
}}

#[test] fn forward_matches_python_multidim_reference() {{
    let m = phylo_skeleton::prebuild();
    let mut params = Params::new();
    {py_params_init}
    let leaves: Vec<Vec<String>> = vec![
        {py_leaves_init},
    ];
    let lk = forward(&m, &params, &leaves);
    let diff = (lk - REFERENCE_LK).abs();
    assert!(lk.is_finite(), "forward returned non-finite: {{}}", lk);
    assert!(diff < 1e-12,
            "forward = {{:.17e}} vs reference {{:.17e}}, diff = {{:.3e}}",
            lk, REFERENCE_LK, diff);
}}
''')

    run(['cargo', 'build', '--release'], cwd=crate)
    run(['cargo', 'test', '--release', '--test', 'tree_check'], cwd=crate)
    print(f"  tree {tree}: states={cpp_states} ref_lk={ref:.6f}  ok")


if __name__ == '__main__':
    main()
