#!/usr/bin/env python3
"""TKF91 verification of the Rust codegen against an exact-lse Python reference.

The default boss build uses an approximate lookup table for log_sum_exp_unary
(LOG_SUM_EXP_LOOKUP_MAX=10), which silently drops contributions whose log-prob
gap exceeds ~10 nats. The emitted Rust uses exact `(lo - hi).exp().ln_1p()`,
so for non-trivial models the Rust Forward is *more accurate* than `boss -L`.

This test compares the codegen Forward against the equivalent multidimensional
Forward DP implemented in Python (with exact log_sum_exp), running on the same
phylo-composed machine. We observe agreement to floating-point noise
(~1e-15) and enforce a 1e-12 tolerance to leave headroom for platform
variation in the order of floating-point operations.
"""

import os, sys, subprocess, json, tempfile, math
sys.path.insert(0, os.path.dirname(__file__))
from _phylo_ref import multidim_forward

REPO = os.environ.get('REPO_ROOT', os.getcwd())
BOSS = os.path.join(REPO, 'bin', 'boss')

def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        sys.stderr.write(f"FAIL: {cmd}\n{r.stderr}\n")
        sys.exit(1)
    return r.stdout

def main():
    workdir = tempfile.mkdtemp(prefix='rust_codegen_tkf91_')
    crate = os.path.join(workdir, 'crate')

    params = {'insRate': 0.005, 'delRate': 0.01,
              'time[A]': 0.3, 'time[B]': 0.2}

    open_args = ['--pair-json',
                 '--preset', 'tkf91-root-dna-jc',
                 '-m', '--begin',
                 '--preset', 'tkf91-branch-dna-jc',
                 '--phylo-tree-string', '(A,B)P;', '--phylo-time-param', 'time',
                 '--end']

    # 1) Codegen
    run([BOSS] + open_args + ['--codegen', crate, '--rust-phylo-hmm'])

    # 2) Reference: exact-lse Python multidim Forward over the same machine
    machine_json = run([BOSS] + open_args)
    leaves = [list('ACGT'), list('ACG')]
    ref = multidim_forward(machine_json, params, leaves)

    # 3) Build & run the Rust crate
    check_rs = os.path.join(crate, 'examples', 'check.rs')
    os.makedirs(os.path.dirname(check_rs), exist_ok=True)
    with open(check_rs, 'w') as f:
        f.write('''use phylo_dp::{forward, viterbi, precompute_log_weights,
                 forward_with_log_weights, viterbi_with_log_weights,
                 Params, ALPHABET};
fn idx(c: char) -> u32 {
    let s = c.to_string();
    ALPHABET.iter().position(|x| *x == s).unwrap() as u32
}
fn main() {
    let p = Params { delRate: 0.01, insRate: 0.005, time_A_: 0.3, time_B_: 0.2 };
    let a: Vec<u32> = "ACGT".chars().map(idx).collect();
    let b: Vec<u32> = "ACG".chars().map(idx).collect();
    let f = forward(&p, [&a, &b]);
    let v = viterbi(&p, [&a, &b]);
    // Amortized API consistency: precompute + *_with_log_weights must
    // give bit-exact equality with the convenience wrappers.
    let lw = precompute_log_weights(&p);
    let f2 = forward_with_log_weights(&lw, [&a, &b]);
    let v2 = viterbi_with_log_weights(&lw, [&a, &b]);
    assert_eq!(f.to_bits(), f2.to_bits(),
        "forward != forward_with_log_weights (f={} f2={})", f, f2);
    assert_eq!(v.to_bits(), v2.to_bits(),
        "viterbi != viterbi_with_log_weights (v={} v2={})", v, v2);
    println!("{} {}", f, v);
}
''')
    out = run(['cargo', 'run', '--release', '--example', 'check', '--quiet'], cwd=crate)
    fwd, vit = (float(x) for x in out.strip().split())

    print(f"forward = {fwd:.15f}")
    print(f"viterbi = {vit:.15f}")
    print(f"ref     = {ref:.15f}  (Python multidim Forward, exact lse)")
    print(f"|fwd-ref| = {abs(fwd-ref):.3e}")

    tol = 1e-12
    fail = False
    if abs(fwd - ref) > tol:
        print(f"FAIL: Rust Forward != exact-lse reference"); fail = True
    if fwd + 1e-12 < vit:
        print(f"FAIL: Forward < Viterbi"); fail = True
    if fail: sys.exit(1)
    print("OK")

if __name__ == '__main__':
    main()
