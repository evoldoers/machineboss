#!/usr/bin/env python3
"""TKF92 + HKY85 quartet — Rust codegen smoke + timing test.

Tree:        (A,B,(C,D)Y)X;
Branches:    t_XA = t[A], t_XB = t[B], t_XY = t[Y], t_YC = t[C], t_YD = t[D]
Substitution: HKY85 (kappa = tsRatio, stationary freqs pi_{A,C,G,T})
Indels:       TKF92 (lambda = insRate, mu = delRate, geometric extension r)

This is the test the user actually asked for: codegen the quartet, build
the emitted Rust crate, and run forward + viterbi on short observed leaves.
A full exact-lse reference is too expensive for the quartet at length>=2
(open-machine has ~213k transitions; Python multidim DP iterating those
once per cell takes minutes), so this test instead:

  1) verifies the codegen pipeline runs end-to-end on the quartet topology;
  2) sanity-checks Forward >= Viterbi and both are finite;
  3) reports wall-clock timings for codegen, Rust compile, and runtime
     across a small sweep of leaf lengths.

The DP code itself is verified independently by check_tkf91.py and
check_tkf92_triad.py against an exact-lse Python reference, both to
floating-point precision.

NOT in default `make test`. Run via:
    REPO_ROOT=$(pwd) python3 t/rust/check_tkf92_quartet.py
or
    make test-rust-codegen-tkf92-quartet
"""

import os, sys, subprocess, json, tempfile, time

REPO = os.environ.get('REPO_ROOT', os.getcwd())
BOSS = os.path.join(REPO, 'bin', 'boss')

def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        sys.stderr.write(f"FAIL: {cmd}\n{r.stderr}\n")
        sys.exit(1)
    return r.stdout

def main():
    tmp = tempfile.mkdtemp(prefix='quartet_')
    crate = os.path.join(tmp, 'crate')

    open_args = [
        '--pair-json',
        '--tkf92-root-dna-hky85',
        '-m', '--begin', '--tkf92-branch-dna-hky85',
        '--phylo-tree-string', '(A,B,(C,D)Y)X;',
        '--end',
    ]

    # Codegen
    t0 = time.time()
    r = subprocess.run([BOSS] + open_args + ['--codegen', crate, '--rust', '-v', '2'],
                       capture_output=True, text=True)
    t_codegen = time.time() - t0
    if r.returncode != 0:
        sys.stderr.write(r.stderr); sys.exit(1)
    print(f"codegen: {t_codegen:.1f}s")
    info_lines = [l for l in r.stderr.splitlines() if 'Wrote Rust crate' in l]
    if info_lines:
        print(f"  {info_lines[-1]}")

    # Build a check binary that runs the DP across a length sweep.
    check_rs = os.path.join(crate, 'examples', 'check.rs')
    os.makedirs(os.path.dirname(check_rs), exist_ok=True)
    with open(check_rs, 'w') as f:
        f.write('''use phylo_dp::{forward, viterbi, Params, ALPHABET};
use std::time::Instant;
fn idx(c: char) -> u32 { ALPHABET.iter().position(|x| x.chars().next() == Some(c)).unwrap() as u32 }
fn mk(n: usize, seed: u64) -> Vec<u32> {
    let mut x = seed;
    (0..n).map(|_| {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((x >> 32) % 4) as u32
    }).collect()
}
fn main() {
    // Tree (A,B,(C,D)Y)X;: branch labels = t_A, t_B, t_Y (X->Y), t_C (Y->C), t_D (Y->D).
    // The unused `t` field comes from a root def that references `t` for HKY85
    // (the def isn't reachable from any transition but currently ends up in Params).
    let p = Params {
        delRate: 0.06, insRate: 0.05, r: 0.4, tsRatio: 2.0,
        pi_A: 0.30, pi_C: 0.20, pi_G: 0.25, pi_T: 0.25,
        t: 0.0,
        t_A_: 0.10, t_B_: 0.20, t_Y_: 0.05, t_C_: 0.15, t_D_: 0.18,
    };
    println!("# leaves of length n at each tip; uniform deterministic random sequences");
    println!("# n   forward(ms)   viterbi(ms)   forward         viterbi");
    for &n in &[2usize, 3, 5, 8, 10] {
        let a = mk(n, 1); let b = mk(n, 2); let c = mk(n, 3); let d = mk(n, 4);
        let t0 = Instant::now();
        let f = forward(&p, [&a, &b, &c, &d]);
        let dt_f = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let v = viterbi(&p, [&a, &b, &c, &d]);
        let dt_v = t0.elapsed().as_secs_f64() * 1000.0;
        println!("{:>3}   {:>10.1}   {:>10.1}   {:.6}   {:.6}", n, dt_f, dt_v, f, v);
        if !f.is_finite() { eprintln!("FAIL: Forward not finite at n={}", n); std::process::exit(1); }
        if !v.is_finite() { eprintln!("FAIL: Viterbi not finite at n={}", n); std::process::exit(1); }
        if f + 1e-9 < v { eprintln!("FAIL: Forward < Viterbi at n={}", n); std::process::exit(1); }
    }
}
''')

    t0 = time.time()
    run(['cargo', 'build', '--release', '--example', 'check', '--quiet'], cwd=crate)
    t_build = time.time() - t0
    print(f"cargo build --release: {t_build:.1f}s")

    out = run(['cargo', 'run', '--release', '--example', 'check', '--quiet'], cwd=crate)
    sys.stdout.write(out)
    print("OK")

if __name__ == '__main__':
    main()
