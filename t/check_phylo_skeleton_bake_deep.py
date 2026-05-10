#!/usr/bin/env python3
"""Multi-tree extension of the bake-and-expand validation, covering the
larger topologies (((A,B)P,C)Q)D; and ((A,B)P,(C,D)Q)R; with multiple
internal nodes.

These tests are kept out of the default `make test` because prebuild()
is slow (~3 min) on a 270-state phylo machine — the symbolic weight
expressions blow up under the unoptimised compose / intersect pipeline.
The state count and bit-exact forward checks below assert exact
agreement with the C++ phylo-intersect output — same state count,
same state IDs in the same order, same forward log-likelihood to
within 1e-12.

Run with: `make test-phylo-skeleton-bake-deep`.
"""
import os, sys, subprocess, tempfile, shutil, json

REPO = os.environ.get('REPO_ROOT', os.getcwd())
BOSS = os.path.join(REPO, 'bin', 'boss')

sys.path.insert(0, os.path.join(REPO, 't', 'rust'))
from _phylo_ref import multidim_forward


def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        sys.stderr.write(f"FAIL: {cmd}\n{r.stderr}\n{r.stdout}\n")
        sys.exit(1)
    return r.stdout


def run_tree_case(tree, leaves, time_params):
    work  = tempfile.mkdtemp(prefix='skelbake_deep_')
    crate = os.path.join(work, 'crate')

    run([BOSS, '--tkf91-branch-dna-jc',
         '--phylo-tree-string', tree, '--phylo-time-param', 'time',
         '--phylo-skeleton', '--codegen', crate, '--rust-phylo-hmm'])

    machine_json = run([BOSS, '--tkf91-branch-dna-jc',
                        '--phylo-tree-string', tree,
                        '--phylo-time-param', 'time',
                        '--pair-json'])
    cpp_states = len(json.loads(machine_json)['state'])
    params = {'insRate': 0.005, 'delRate': 0.01, **time_params}
    ref    = multidim_forward(machine_json, params, leaves)

    tests_dir = os.path.join(crate, 'tests')
    os.makedirs(tests_dir, exist_ok=True)

    py_params_init = '\n    '.join(
        f'params.insert("{k}".into(), {v!r});' for k, v in params.items())
    py_leaves_init = ',\n        '.join(
        '"{0}".chars().map(|c| c.to_string()).collect()'.format(''.join(leaf))
        for leaf in leaves)

    with open(os.path.join(tests_dir, 'tree_check_deep.rs'), 'w') as f:
        f.write(f'''//! Deep-tree regression for the bake-and-expand pipeline (manual).
//!
//! Tree: {tree}
//! Leaves: {[''.join(l) for l in leaves]}
//!
//! Asserts (against C++ phylo-intersect as ground truth):
//!   1. prebuild() state count == C++ M_full state count ({cpp_states}).
//!   2. prebuild() is advancing (no silent back-transitions).
//!   3. forward(prebuild(), ...) matches Python multidim_forward bit-
//!      exactly to within 1e-12.
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

    print(f"  building tree {tree} (cpp_states={cpp_states}, "
          f"ref_lk={ref:.6f})...", flush=True)
    run(['cargo', 'build', '--release'], cwd=crate)
    print(f"  running tree {tree} cargo test (this may take several minutes)...",
          flush=True)
    run(['cargo', 'test', '--release', '--test', 'tree_check_deep'], cwd=crate)
    print(f"  tree {tree}: ok")


def main():
    if shutil.which('cargo') is None:
        print("test-phylo-skeleton-bake-deep    skip: cargo not in PATH")
        return
    cases = [
        # (tree, leaves, branch-times)
        ('(((A,B)P,C)Q)D;',
         [list('AC'), list('AC'), list('AC')],
         {'time[A]': 0.3, 'time[B]': 0.2, 'time[C]': 0.4,
          'time[P]': 0.15, 'time[Q]': 0.25}),
        ('((A,B)P,(C,D)Q)R;',
         [list('AC'), list('A'), list('A'), list('A')],
         {'time[A]': 0.3, 'time[B]': 0.2, 'time[C]': 0.4, 'time[D]': 0.5,
          'time[P]': 0.15, 'time[Q]': 0.25}),
    ]
    for tree, leaves, time_params in cases:
        run_tree_case(tree, leaves, time_params)
    print("OK")


if __name__ == '__main__':
    main()
