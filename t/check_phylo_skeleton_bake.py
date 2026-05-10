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

    # Drop a tests/ file that asserts the four constants and the panic.
    tests_dir = os.path.join(crate, 'tests')
    os.makedirs(tests_dir, exist_ok=True)
    with open(os.path.join(tests_dir, 'bake_parses.rs'), 'w') as f:
        f.write('''use phylo_skeleton::{T_JSON, M_SKEL_JSON, TREE_NEWICK, TIME_PARAM};
#[test] fn t_json_parses() {
    let v: serde_json::Value = serde_json::from_str(T_JSON).expect("T_JSON parses");
    assert!(v.get("state").is_some());
}
#[test] fn m_skel_json_parses() {
    let v: serde_json::Value = serde_json::from_str(M_SKEL_JSON).expect("M_SKEL_JSON parses");
    assert!(v.get("state").is_some());
}
#[test] fn tree_newick_well_formed() {
    assert!(TREE_NEWICK.ends_with(";"));
    assert_eq!(TREE_NEWICK.matches('(').count(), TREE_NEWICK.matches(')').count());
}
#[test] fn time_param_nonempty() { assert!(!TIME_PARAM.is_empty()); }
#[test] #[should_panic(expected = "not yet implemented")]
fn prebuild_panics() { phylo_skeleton::prebuild(); }
''')

    run(['cargo', 'build', '--release'], cwd=crate)
    run(['cargo', 'test', '--release'], cwd=crate)
    print("OK")

if __name__ == '__main__':
    main()
