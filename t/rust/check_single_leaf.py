#!/usr/bin/env python3
"""Regression: codegen handles L=1 (single-leaf and chain) phylo trees.

When the tree has only one leaf, no `intersect` happens during the phylo
build, so output tokens are bare symbols (e.g. `A`) rather than JSON-
encoded pair tokens. The codegen's pair-token parser must accept that.

Verifies two L=1 topologies — `(A)X;` (single branch) and `((A)Y)X;`
(chain via degree-1 internal node) — against the exact-lse Python multidim
Forward reference.
"""

import os, sys, subprocess, tempfile, json
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

CASES = [
    {
        'name': 'single (A)X;',
        'tree': '(A)X;',
        'params': {'insRate': 0.005, 'delRate': 0.01, 'time[A]': 0.3},
        'rust_params': 'Params { delRate: 0.01, insRate: 0.005, time_A_: 0.3 }',
        'leaf': 'ACGT',
    },
    {
        'name': 'chain ((A)Y)X;',
        'tree': '((A)Y)X;',
        'params': {'insRate': 0.005, 'delRate': 0.01,
                   'time[A]': 0.2, 'time[Y]': 0.15},
        'rust_params': 'Params { delRate: 0.01, insRate: 0.005, time_A_: 0.2, time_Y_: 0.15 }',
        'leaf': 'ACG',
    },
]

def main():
    for case in CASES:
        print(f"--- {case['name']} ---")
        workdir = tempfile.mkdtemp(prefix='single_')
        crate = os.path.join(workdir, 'crate')

        open_args = ['--pair-json', '--tkf91-root-dna-jc',
                     '-m', '--begin', '--tkf91-branch-dna-jc',
                     '--phylo-tree-string', case['tree'],
                     '--phylo-time-param', 'time', '--end']

        run([BOSS] + open_args + ['--codegen', crate, '--rust'])
        machine_json = run([BOSS] + open_args)
        ref = multidim_forward(machine_json, case['params'], [list(case['leaf'])])

        check_rs = os.path.join(crate, 'examples', 'check.rs')
        os.makedirs(os.path.dirname(check_rs), exist_ok=True)
        with open(check_rs, 'w') as f:
            f.write(f'''use phylo_dp::{{forward, Params, ALPHABET}};
fn idx(c: char) -> u32 {{ ALPHABET.iter().position(|x| x.chars().next() == Some(c)).unwrap() as u32 }}
fn main() {{
    let p = {case['rust_params']};
    let a: Vec<u32> = "{case['leaf']}".chars().map(idx).collect();
    println!("{{}}", forward(&p, [&a]));
}}
''')
        run(['cargo', 'build', '--release', '--example', 'check', '--quiet'], cwd=crate)
        out = run(['cargo', 'run', '--release', '--example', 'check', '--quiet'], cwd=crate).strip()
        fwd = float(out)

        print(f"  forward = {fwd:.15f}")
        print(f"  ref     = {ref:.15f}")
        print(f"  |fwd-ref| = {abs(fwd-ref):.3e}")
        if abs(fwd - ref) > 1e-9:
            print("  FAIL"); sys.exit(1)
    print("OK")

if __name__ == '__main__':
    main()
