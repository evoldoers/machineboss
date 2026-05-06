#!/usr/bin/env python3
"""TKF91 verification of the Rust codegen against an exact-lse reference.

The default boss build uses an approximate lookup table for log_sum_exp_unary
(LOG_SUM_EXP_LOOKUP_MAX=10), which silently drops contributions whose log-prob
gap exceeds ~10 nats. The emitted Rust uses exact `(lo - hi).exp().ln_1p()`,
so for non-trivial models the Rust Forward is *more accurate* than boss -L
under the default build.

This test computes the marginal P(observed leaves) two ways and asserts
floating-point agreement:

  (1) Codegen the open phylo machine, build & run the emitted Rust crate.
  (2) Build the same model clamped at the leaves (via --phylo-clamp), then
      run an exact-lse Forward implemented in Python on the dumped JSON.
"""

import os, sys, subprocess, json, tempfile, math

REPO = os.environ.get('REPO_ROOT', os.getcwd())
BOSS = os.path.join(REPO, 'bin', 'boss')

def run(cmd, cwd=None, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, **kw)
    if r.returncode != 0:
        sys.stderr.write(f"FAIL: {cmd}\n{r.stderr}\n")
        sys.exit(1)
    return r.stdout

def lse(a, b):
    if a == -math.inf: return b
    if b == -math.inf: return a
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))

def eval_w(w, p):
    if isinstance(w, (int, float)): return float(w)
    if isinstance(w, str): return p[w]
    if isinstance(w, list): return math.prod(eval_w(a, p) for a in w)
    if isinstance(w, dict):
        for op, args in w.items():
            if op == '*': return math.prod(eval_w(a, p) for a in args)
            if op == '/': return eval_w(args[0], p) / eval_w(args[1], p)
            if op == '+': return sum(eval_w(a, p) for a in args)
            if op == '-': return eval_w(args[0], p) - eval_w(args[1], p)
            if op == 'log': return math.log(eval_w(args, p))
            if op == 'exp': return math.exp(eval_w(args, p))
            if op == 'not': return 1.0 - eval_w(args, p)
            if op == 'pow': return eval_w(args[0], p) ** eval_w(args[1], p)
            if op == 'geomsum': return 1.0 / (1.0 - eval_w(args, p))
            raise ValueError(f"unknown op: {op}")
    raise ValueError(f"weight: {w}")

def expand_defs(defs, params):
    full = dict(params)
    for _ in range(100):
        ch = False
        for k, v in defs.items():
            try:
                vv = eval_w(v, full)
                if k not in full or full[k] != vv:
                    full[k] = vv; ch = True
            except (KeyError, ZeroDivisionError, TypeError):
                pass
        if not ch: break
    return full

def exact_forward(machine_json, params):
    """Forward over a no-input/no-output advance-sorted Markov chain.
    Uses exact log_sum_exp via math.log1p(math.exp(.))."""
    m = json.loads(machine_json)
    full = expand_defs(m.get('defs', {}), params)
    states = m['state']
    N = len(states)
    f = [-math.inf] * N
    f[0] = 0.0
    for s_idx, s in enumerate(states):
        if f[s_idx] == -math.inf: continue
        for t in s.get('trans', []):
            w = eval_w(t.get('weight', 1), full)
            if w <= 0: continue
            d = t['to']
            f[d] = lse(f[d], f[s_idx] + math.log(w))
    return f[N - 1]

def main():
    workdir = tempfile.mkdtemp(prefix='rust_codegen_tkf91_')
    crate = os.path.join(workdir, 'crate')
    clamp_f = os.path.join(workdir, 'clamp.json')
    params_f = os.path.join(workdir, 'params.json')

    params = {'insRate': 0.005, 'delRate': 0.01,
              'time[A]': 0.3, 'time[B]': 0.2}
    with open(clamp_f, 'w') as f:
        json.dump({'A': ['A','C','G','T'], 'B': ['A','C','G']}, f)
    with open(params_f, 'w') as f:
        json.dump(params, f)

    open_args = ['--pair-json', '--preset', 'tkf91-root-dna-jc',
                 '-m', '--begin', '--preset', 'tkf91-branch-dna-jc',
                 '--phylo-tree-string', '(A,B)P;', '--phylo-time-param', 'time',
                 '--end']
    clamped_args = open_args[:-1] + ['--phylo-clamp', clamp_f, '--end']

    # 1) Codegen the open machine
    run([BOSS] + open_args + ['--codegen', crate, '--rust'])

    # 2) Reference: exact Forward over the clamped machine, in Python
    clamped_json = run([BOSS] + clamped_args)
    ref = exact_forward(clamped_json, params)

    # 3) Build & run the Rust crate
    check_rs = os.path.join(crate, 'examples', 'check.rs')
    os.makedirs(os.path.dirname(check_rs), exist_ok=True)
    with open(check_rs, 'w') as f:
        f.write('''use phylo_dp::{forward, viterbi, Params, ALPHABET};
fn idx(c: char) -> u32 {
    let s = c.to_string();
    ALPHABET.iter().position(|x| *x == s).unwrap() as u32
}
fn main() {
    let p = Params { delRate: 0.01, insRate: 0.005, time_A_: 0.3, time_B_: 0.2 };
    let a: Vec<u32> = "ACGT".chars().map(idx).collect();
    let b: Vec<u32> = "ACG".chars().map(idx).collect();
    println!("{} {}", forward(&p, [&a, &b]), viterbi(&p, [&a, &b]));
}
''')
    out = run(['cargo', 'run', '--release', '--example', 'check', '--quiet'], cwd=crate)
    fwd, vit = (float(x) for x in out.strip().split())

    print(f"forward = {fwd:.15f}")
    print(f"viterbi = {vit:.15f}")
    print(f"ref     = {ref:.15f}  (Python exact Forward over --phylo-clamp)")
    print(f"|fwd-ref| = {abs(fwd-ref):.3e}")

    tol = 1e-9
    fail = False
    if abs(fwd - ref) > tol:
        print(f"FAIL: Rust Forward != exact-lse reference"); fail = True
    if fwd + 1e-12 < vit:
        print(f"FAIL: Forward < Viterbi (logsumexp must be >= max)"); fail = True
    if fail: sys.exit(1)
    print("OK")

if __name__ == '__main__':
    main()
