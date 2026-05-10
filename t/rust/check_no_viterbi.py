#!/usr/bin/env python3
"""Regression: --no-viterbi cleanly omits the Viterbi function pair.

Codegens an echo phylo with --no-viterbi, asserts that:
  - the emitted lib.rs has `pub fn forward` and `pub fn forward_with_log_weights`
  - it does NOT have `pub fn viterbi` or `pub fn viterbi_with_log_weights`
  - the crate still compiles
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
    workdir = tempfile.mkdtemp(prefix='no_viterbi_')
    crate = os.path.join(workdir, 'crate')
    echo = os.path.join(REPO, 't', 'machine', 'echo-with-time.json')

    run([BOSS, '--pair-json', '--generate-uniform', '01', '-m',
         '--begin', echo, '--phylo-tree-string', '(A,B)P;', '--end',
         '--codegen', crate, '--rust-phylo-hmm', '--no-viterbi'])

    with open(os.path.join(crate, 'src', 'lib.rs')) as f:
        lib = f.read()

    must_have = ['pub fn forward(', 'pub fn forward_with_log_weights(']
    must_not_have = ['pub fn viterbi(', 'pub fn viterbi_with_log_weights(']

    fail = False
    for needle in must_have:
        if needle not in lib:
            print(f"FAIL: missing required `{needle}` in lib.rs"); fail = True
    for needle in must_not_have:
        if needle in lib:
            print(f"FAIL: --no-viterbi did not strip `{needle}`"); fail = True
    if fail: sys.exit(1)

    # The crate must still compile.
    run(['cargo', 'build', '--release', '--quiet'], cwd=crate)

    print("OK")

if __name__ == '__main__':
    main()
