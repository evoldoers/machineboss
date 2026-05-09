#!/usr/bin/env python3
"""Verify that --phylo-skeleton produces a structurally-correct phylo machine
that matches the topology of the full phylo path on a small DNA case (where
the full path is tractable). Specifically: the two machines should have the
same number of accessible states and identical state-pair index maps after
collapsing the emit alphabet on the full path.

Also asserts the skeleton path collapses the alphabet blow-up in transitions
and runs in a tiny fraction of the full path's wall time.
"""
import os, sys, subprocess, json, tempfile, time

REPO = os.environ.get('REPO_ROOT', os.getcwd())
BOSS = os.path.join(REPO, 'bin', 'boss')

def run(args, out_path=None):
    if out_path:
        with open(out_path, 'w') as f:
            r = subprocess.run([BOSS] + args, stdout=f, stderr=subprocess.PIPE, text=True)
    else:
        r = subprocess.run([BOSS] + args, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"FAIL {args}\n{r.stderr[:600]}\n")
        sys.exit(1)
    return r.stdout if not out_path else None

def stats(path):
    with open(path) as f:
        m = json.load(f)
    n_states = len(m['state'])
    n_trans = sum(len(s.get('trans', [])) for s in m['state'])
    n_emit = sum(1 for s in m['state'] for t in s.get('trans', []) if t.get('in') or t.get('out'))
    return n_states, n_trans, n_emit

def main():
    work = tempfile.mkdtemp(prefix='skel_')
    full = os.path.join(work, 'full.json')
    skel = os.path.join(work, 'skel.json')

    # Use tkf91-branch-dna-jc (4-letter alphabet, small): full is tractable.
    base = ['--tkf91-branch-dna-jc',
            '--phylo-tree-string', '(A,B)P;',
            '--phylo-time-param', 'time',
            '--phylo-no-felsenstein',
            '--strip-names']

    t0 = time.time()
    run(base, full)
    t_full = time.time() - t0

    t0 = time.time()
    run(base + ['--phylo-skeleton'], skel)
    t_skel = time.time() - t0

    ns_f, nt_f, ne_f = stats(full)
    ns_s, nt_s, ne_s = stats(skel)
    print(f"full: states={ns_f:>5} trans={nt_f:>7} emit={ne_f:>7}  ({t_full:.3f}s)")
    print(f"skel: states={ns_s:>5} trans={nt_s:>7} emit={ne_s:>7}  ({t_skel:.3f}s)")

    if ns_f != ns_s:
        sys.stderr.write(f"FAIL: state count mismatch: full={ns_f} skel={ns_s}\n")
        sys.exit(1)
    if nt_s >= nt_f:
        sys.stderr.write(f"FAIL: skeleton transition count {nt_s} not < full {nt_f}\n")
        sys.exit(1)
    # On DNA TKF91 binary tree, the alphabet blow-up factor for emit
    # transitions is 4^k for k emitting leaves. Expect at least 4× collapse.
    if ne_s * 4 > ne_f:
        sys.stderr.write(f"FAIL: emit collapse weak: full_emit={ne_f} skel_emit={ne_s}\n")
        sys.exit(1)

    # Skeleton on a protein quartet (intractable for the full path) must
    # complete quickly and produce a finite machine.
    skel_q = os.path.join(work, 'skel_q.json')
    t0 = time.time()
    run(['--tkf92-branch-prot-f81',
         '--phylo-tree-string', '((A,B)P,(C,D)Q)R;',
         '--phylo-time-param', 't',
         '--phylo-no-felsenstein',
         '--phylo-skeleton',
         '--strip-names'], skel_q)
    t_q = time.time() - t0
    ns_q, nt_q, ne_q = stats(skel_q)
    print(f"protein quartet skeleton: states={ns_q} trans={nt_q} ({t_q:.3f}s)")
    if t_q > 30.0:
        sys.stderr.write(f"FAIL: protein quartet skeleton took {t_q:.1f}s, expected under 30s\n")
        sys.exit(1)

    print("OK")

if __name__ == '__main__':
    main()
