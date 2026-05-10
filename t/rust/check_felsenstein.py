#!/usr/bin/env python3
"""Verify that the Felsenstein-hoisted phylo machine and the
transition-exploded one give the same Forward log-likelihood (bit-exactly,
since boss -L uses the same lookup-table approximation in both modes).

Also report the size differences so future maintainers can see whether the
hoisting is still earning its keep.
"""

import os, sys, subprocess, tempfile, json

REPO = os.environ.get('REPO_ROOT', os.getcwd())
BOSS = os.path.join(REPO, 'bin', 'boss')

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"FAIL: {cmd}\n{r.stderr}\n")
        sys.exit(1)
    return r.stdout

def main():
    workdir = tempfile.mkdtemp(prefix='fels_')
    clamp = os.path.join(workdir, 'clamp.json')
    params = os.path.join(workdir, 'params.json')
    with open(clamp, 'w') as f:
        json.dump({'A': ['A','C','G','T'], 'B': ['A','C','G']}, f)
    with open(params, 'w') as f:
        json.dump({'insRate': 0.005, 'delRate': 0.01,
                   'time[A]': 0.3, 'time[B]': 0.2}, f)

    base = ['--pair-json', '--tkf91-root-dna-jc',
            '-m', '--begin', '--tkf91-branch-dna-jc',
            '--phylo-tree-string', '(A,B)P;', '--phylo-time-param', 'time',
            '--phylo-clamp', clamp, '--end',
            '-P', params, '-L']

    out_fels = run([BOSS] + base)
    out_nofels = run([BOSS] + base + ['--phylo-no-felsenstein'])
    ll_fels = json.loads(out_fels)[0][2]
    ll_nofels = json.loads(out_nofels)[0][2]
    print(f"-L felsenstein:    {ll_fels}")
    print(f"-L no-felsenstein: {ll_nofels}")
    if ll_fels != ll_nofels:
        print(f"FAIL: log-likelihoods differ"); sys.exit(1)

    # Also report machine.json size for a non-trivial protein case.
    crate_fels   = os.path.join(workdir, 'p_fels')
    crate_nofels = os.path.join(workdir, 'p_nofels')
    open_args = ['--pair-json', '--tkf92-root-prot-f81',
                 '-m', '--begin', '--tkf92-branch-prot-f81',
                 '--phylo-tree-string', '(A,B)P;', '--end']
    run([BOSS] + open_args + ['--codegen', crate_fels, '--rust-phylo-hmm'])
    run([BOSS] + open_args + ['--phylo-no-felsenstein',
                              '--codegen', crate_nofels, '--rust-phylo-hmm'])
    sz_lib_fels   = os.path.getsize(os.path.join(crate_fels,   'src/lib.rs'))
    sz_lib_nofels = os.path.getsize(os.path.join(crate_nofels, 'src/lib.rs'))
    sz_mj_fels    = os.path.getsize(os.path.join(crate_fels,   'machine.json'))
    sz_mj_nofels  = os.path.getsize(os.path.join(crate_nofels, 'machine.json'))
    print(f"protein binary tree, lib.rs:      fels={sz_lib_fels:>10}  nofels={sz_lib_nofels:>10}  ({100*sz_lib_fels/sz_lib_nofels:.0f}%)")
    print(f"protein binary tree, machine.json: fels={sz_mj_fels:>10}  nofels={sz_mj_nofels:>10}  ({100*sz_mj_fels/sz_mj_nofels:.0f}%)")
    if sz_lib_fels >= sz_lib_nofels:
        print(f"WARN: hoisting did not shrink lib.rs")
    if sz_mj_fels >= sz_mj_nofels:
        print(f"WARN: hoisting did not shrink machine.json")
    print("OK")

if __name__ == '__main__':
    main()
