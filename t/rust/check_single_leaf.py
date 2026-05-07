#!/usr/bin/env python3
"""Regression: codegen handles L=1 (single-leaf) phylo trees.

When the tree has only one leaf, no `intersect` happens during the phylo
build, so output tokens are bare symbols (e.g. `A`) rather than JSON-
encoded pair tokens. The codegen's pair-token parser must accept that.

Verifies the L=1 codegen against `boss --phylo-clamp -L` (allowing for
boss's log_sum_exp lookup-table tolerance).
"""

import os, sys, subprocess, tempfile, json

REPO = os.environ.get('REPO_ROOT', os.getcwd())
BOSS = os.path.join(REPO, 'bin', 'boss')

def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        sys.stderr.write(f"FAIL: {cmd}\n{r.stderr}\n")
        sys.exit(1)
    return r.stdout

def main():
    workdir = tempfile.mkdtemp(prefix='single_leaf_')
    crate = os.path.join(workdir, 'crate')
    clamp = os.path.join(workdir, 'clamp.json')
    params = os.path.join(workdir, 'params.json')
    with open(clamp, 'w') as f: json.dump({'A': ['A', 'C', 'G', 'T']}, f)
    with open(params, 'w') as f:
        json.dump({'insRate': 0.005, 'delRate': 0.01, 'time[A]': 0.3}, f)

    open_args = ['--pair-json', '--tkf91-root-dna-jc',
                 '-m', '--begin', '--tkf91-branch-dna-jc',
                 '--phylo-tree-string', '(A)X;', '--phylo-time-param', 'time',
                 '--end']

    run([BOSS] + open_args + ['--codegen', crate, '--rust'])

    out = run([BOSS] + open_args[:-1] + ['--phylo-clamp', clamp, '--end',
                                          '-P', params, '-L'])
    ref = json.loads(out)[0][2]

    check_rs = os.path.join(crate, 'examples', 'check.rs')
    os.makedirs(os.path.dirname(check_rs), exist_ok=True)
    with open(check_rs, 'w') as f:
        f.write('''use phylo_dp::{forward, Params, ALPHABET};
fn idx(c: char) -> u32 { ALPHABET.iter().position(|x| x.chars().next() == Some(c)).unwrap() as u32 }
fn main() {
    let p = Params { delRate: 0.01, insRate: 0.005, time_A_: 0.3 };
    let a: Vec<u32> = "ACGT".chars().map(idx).collect();
    println!("{}", forward(&p, [&a]));
}
''')
    run(['cargo', 'build', '--release', '--example', 'check', '--quiet'], cwd=crate)
    out = run(['cargo', 'run', '--release', '--example', 'check', '--quiet'], cwd=crate).strip()
    fwd = float(out)

    print(f"forward = {fwd:.15f}")
    print(f"boss -L = {ref:.15f}")
    # boss -L's log_sum_exp lookup table introduces ~1e-11 error; codegen is exact.
    if abs(fwd - ref) > 1e-9:
        print(f"FAIL: |fwd - boss-L| = {abs(fwd-ref):.3e} > 1e-9"); sys.exit(1)
    print("OK")

if __name__ == '__main__':
    main()
