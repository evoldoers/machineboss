#!/usr/bin/env python3
"""Verify the Rust codegen Forward / Viterbi against an independent
reference. Strategy:
  1) Codegen a Rust crate from a phylo-composed echo machine.
  2) Build the same machine clamped at the leaves (via --phylo-clamp).
  3) Run boss -L on the clamped machine to get a reference log-likelihood.
  4) Compile and run the codegen Rust forward/viterbi.
  5) Assert all three agree.

For the trivial echo branch transducer there is exactly one column-emission
path consistent with any pair of equal leaves, so Forward = Viterbi =
clamped Forward to floating-point precision."""

import os, sys, subprocess, json, tempfile, math

REPO = os.environ.get('REPO_ROOT', os.getcwd())
BOSS = os.path.join(REPO, 'bin', 'boss')

def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.stderr.write(f"FAIL: {cmd}\n{r.stderr}\n")
        sys.exit(1)
    return r.stdout

def main():
    workdir = tempfile.mkdtemp(prefix='rust_codegen_check_')
    crate = os.path.join(workdir, 'crate')
    clamp = os.path.join(workdir, 'clamp.json')
    params = os.path.join(workdir, 'params.json')
    with open(clamp, 'w') as f:
        json.dump({'A': ['0', '1'], 'B': ['0', '1']}, f)
    with open(params, 'w') as f:
        json.dump({'t[A]': 0.4, 't[B]': 0.7}, f)

    echo_machine = os.path.join(REPO, 't', 'machine', 'echo-with-time.json')
    tree = '(A,B)P;'

    # 1) Codegen
    run([BOSS, '--pair-json', '--generate-uniform', '01', '-m',
         '--begin', echo_machine, '--phylo-tree-string', tree, '--end',
         '--codegen', crate, '--rust-phylo-hmm'])

    # 2) M_clamped reference
    out = run([BOSS, '--pair-json', '--generate-uniform', '01', '-m',
               '--begin', echo_machine, '--phylo-tree-string', tree,
               '--phylo-clamp', clamp, '--end',
               '-P', params, '-L'])
    ref = json.loads(out)[0][2]

    # 4) Build and run the crate
    check_rs = os.path.join(crate, 'examples', 'check.rs')
    os.makedirs(os.path.dirname(check_rs), exist_ok=True)
    with open(check_rs, 'w') as f:
        f.write('''use phylo_dp::{forward, viterbi, Params};
fn main() {
    let p = Params { t_A_: 0.4, t_B_: 0.7 };
    let a: Vec<u32> = vec![0, 1];
    let b: Vec<u32> = vec![0, 1];
    let f = forward(&p, [&a, &b]);
    let v = viterbi(&p, [&a, &b]);
    println!("{} {}", f, v);
}
''')
    out = run(['cargo', 'run', '--release', '--example', 'check', '--quiet'], cwd=crate)
    fwd, vit = (float(x) for x in out.strip().split())

    # Assert agreement (echo has a single feasible path -> all three equal)
    tol = 1e-9
    ok_fv = abs(fwd - vit) < tol
    ok_fr = abs(fwd - ref) < tol
    ok_vr = abs(vit - ref) < tol
    print(f"forward={fwd:.15f}, viterbi={vit:.15f}, ref={ref:.15f}")
    print(f"|fwd-vit|={abs(fwd-vit):.3g}  |fwd-ref|={abs(fwd-ref):.3g}  |vit-ref|={abs(vit-ref):.3g}")
    if not (ok_fv and ok_fr and ok_vr):
        print("MISMATCH"); sys.exit(1)
    # Forward must be >= Viterbi (logsumexp >= max)
    if fwd + 1e-12 < vit:
        print(f"Forward < Viterbi (impossible)"); sys.exit(1)
    print("OK")

if __name__ == '__main__':
    main()
