#!/usr/bin/env python3
"""Validate `boss --rust-transducer` against `boss -L` on a regular
in/out transducer (TKF91-branch-dna-jc).

Generates a Rust crate via `--codegen DIR --rust-transducer`, builds
it, runs forward(p, input, output), and asserts agreement (within a
tight tolerance) with boss's own Forward DP. Also asserts viterbi ≤
forward.

Skipped when cargo is not in PATH.
"""
import os, sys, subprocess, tempfile, shutil, json, math

REPO = os.environ.get('REPO_ROOT', os.getcwd())
BOSS = os.path.join(REPO, 'bin', 'boss')

def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        sys.stderr.write(f"FAIL: {cmd}\n{r.stderr}\n{r.stdout}\n")
        sys.exit(1)
    return r.stdout

def main():
    if shutil.which('cargo') is None:
        print("test-rust-transducer    skip: cargo not in PATH")
        return
    work = tempfile.mkdtemp(prefix='rust_transducer_')
    crate = os.path.join(work, 'crate')

    # 1) Codegen.
    run([BOSS, '--tkf91-branch-dna-jc',
         '--codegen', crate, '--rust-transducer'])

    # 2) Reference: boss's own Forward DP at the same params + sequences.
    params = {'time': 0.1, 'insRate': 0.01, 'delRate': 0.02}
    inp, out = 'ACGT', 'ACGA'
    pf = os.path.join(work, 'params.json')
    with open(pf, 'w') as f: json.dump(params, f)
    boss_lk_raw = run([BOSS, '--tkf91-branch-dna-jc', '-P', pf,
                       '--input-chars', inp, '--output-chars', out, '-L'])
    ref = json.loads(boss_lk_raw)[0][2]    # [["ACGT","ACGA",-3.9736...]]

    # 3) Build & run the Rust crate.
    examples = os.path.join(crate, 'examples')
    os.makedirs(examples, exist_ok=True)
    py_params = '\n    '.join(
        f'p.insert("{k}".into(), {v!r});' for k, v in params.items())
    inp_lit = ', '.join(f'"{c}"' for c in inp)
    out_lit = ', '.join(f'"{c}"' for c in out)
    with open(os.path.join(examples, 'check.rs'), 'w') as f:
        f.write(f'''use transducer_dp::{{forward, viterbi, Params}};
fn main() {{
    let mut p = Params::new();
    {py_params}
    let input  = [{inp_lit}];
    let output = [{out_lit}];
    let f = forward(&p, &input, &output);
    let v = viterbi(&p, &input, &output);
    println!("{{}} {{}}", f, v);
}}
''')
    out_str = run(['cargo', 'run', '--release', '--example', 'check', '--quiet'],
                  cwd=crate)
    fwd, vit = (float(x) for x in out_str.strip().split())

    print(f"forward = {fwd:.15f}")
    print(f"viterbi = {vit:.15f}")
    print(f"ref     = {ref:.15f}  (boss --tkf91-branch-dna-jc -L)")
    print(f"|fwd-ref| = {abs(fwd-ref):.3e}")

    fail = False
    # Boss's Forward uses a lookup-table lse approximation; the Rust crate
    # uses exact lse. The two agree to ~1e-4 in practice. We enforce 1e-3.
    if abs(fwd - ref) > 1e-3:
        print(f"FAIL: Rust forward != boss -L ({abs(fwd-ref):.3e} > 1e-3)")
        fail = True
    if vit > fwd + 1e-12:
        print(f"FAIL: Viterbi > Forward ({vit} > {fwd})")
        fail = True
    if not math.isfinite(fwd):
        print("FAIL: Forward not finite")
        fail = True
    if fail:
        sys.exit(1)
    print("OK")

if __name__ == '__main__':
    main()
