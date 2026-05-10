#!/usr/bin/env python3
"""TKF92 + HKY85 triad — full Forward reference comparison.

Tree: (A,B,C)X;
This is a smaller cousin of the quartet test: same model (TKF92+HKY85 root
+ branch transducer with phylo composition) but only 3 leaves, so the open
machine is small enough that an exact-lse Python multidim Forward DP runs
in seconds and serves as a reference. Verifies the Rust codegen Forward to
floating-point precision.
"""

import os, sys, subprocess, json, tempfile
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
    tmp = tempfile.mkdtemp(prefix='triad_')
    crate = os.path.join(tmp, 'crate')

    leaves = {'A': list('ACG'), 'B': list('AC'), 'C': list('ACT')}
    params = {
        'insRate': 0.05, 'delRate': 0.06, 'r': 0.4,
        'tsRatio': 2.0,
        'pi_A': 0.30, 'pi_C': 0.20, 'pi_G': 0.25, 'pi_T': 0.25,
        't[A]': 0.10, 't[B]': 0.20, 't[C]': 0.15,
    }

    open_args = ['--pair-json', '--tkf92-root-dna-hky85',
                 '-m', '--begin', '--tkf92-branch-dna-hky85',
                 '--phylo-tree-string', '(A,B,C)X;', '--end']

    print("[1/4] codegen ...", file=sys.stderr)
    run([BOSS] + open_args + ['--codegen', crate, '--rust-phylo-hmm'])
    print("[2/4] open machine JSON ...", file=sys.stderr)
    machine_json = run([BOSS] + open_args)
    print("[3/4] Python multidim Forward (reference) ...", file=sys.stderr)
    leaf_strs = [leaves[k] for k in ['A', 'B', 'C']]
    ref = multidim_forward(machine_json, params, leaf_strs)
    print("[4/4] cargo build & run ...", file=sys.stderr)

    check_rs = os.path.join(crate, 'examples', 'check.rs')
    os.makedirs(os.path.dirname(check_rs), exist_ok=True)
    with open(check_rs, 'w') as f:
        f.write('''use phylo_dp::{forward, viterbi, Params, ALPHABET};
fn idx(c: char) -> u32 { ALPHABET.iter().position(|x| x.chars().next() == Some(c)).unwrap() as u32 }
fn main() {
    let p = Params {
        delRate: 0.06, insRate: 0.05, r: 0.4, tsRatio: 2.0,
        pi_A: 0.30, pi_C: 0.20, pi_G: 0.25, pi_T: 0.25,
        t_A_: 0.10, t_B_: 0.20, t_C_: 0.15,
    };
    let a: Vec<u32> = "ACG".chars().map(idx).collect();
    let b: Vec<u32> = "AC".chars().map(idx).collect();
    let c: Vec<u32> = "ACT".chars().map(idx).collect();
    println!("{} {}", forward(&p, [&a, &b, &c]), viterbi(&p, [&a, &b, &c]));
}
''')
    run(['cargo', 'build', '--release', '--example', 'check', '--quiet'], cwd=crate)
    out = run(['cargo', 'run', '--release', '--example', 'check', '--quiet'], cwd=crate).strip()
    fwd, vit = (float(x) for x in out.split())

    print(f"forward = {fwd:.15f}")
    print(f"viterbi = {vit:.15f}")
    print(f"ref     = {ref:.15f}")
    print(f"|fwd-ref| = {abs(fwd - ref):.3e}")

    if abs(fwd - ref) > 1e-12: sys.exit("FAIL: Rust Forward != reference")
    if fwd + 1e-12 < vit: sys.exit("FAIL: Forward < Viterbi")
    print("OK")

if __name__ == '__main__':
    main()
