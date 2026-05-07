#!/usr/bin/env python3
"""TKF92 + HKY85 triad — full Forward reference comparison.

Tree: (A,B,C)X;
This is a smaller cousin of the quartet test: same model (TKF92+HKY85 root
+ branch transducer with phylo composition) but only 3 leaves, so the open
machine is small enough that an exact-lse Python multidim Forward DP runs
in seconds and serves as a reference. Verifies the Rust codegen Forward to
floating-point precision.
"""

import os, sys, subprocess, json, tempfile, math

REPO = os.environ.get('REPO_ROOT', os.getcwd())
BOSS = os.path.join(REPO, 'bin', 'boss')

def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
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
    raise ValueError(w)

def expand(defs, params):
    full = dict(params)
    for _ in range(200):
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

def parse_tok(tok):
    return "" if not tok else json.loads(tok)

def merge_shape(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return [merge_shape(x, y) for x, y in zip(a, b)]
    if isinstance(a, list): return a
    if isinstance(b, list): return b
    return ""

def count_leaves(t):
    return 1 if not isinstance(t, list) else sum(count_leaves(c) for c in t)

def decode(tok, tmpl, out):
    if not isinstance(tmpl, list):
        out.append(None if tok == "" else tok); return
    if not isinstance(tok, list):
        if tok == "":
            for _ in range(count_leaves(tmpl)):
                out.append(None)
            return
        raise ValueError("non-empty leaf where array expected")
    for ti, ts in zip(tok, tmpl):
        decode(ti, ts, out)

def multidim_forward(machine_json, params, leaves):
    m = json.loads(machine_json)
    full = expand(m.get('defs', {}), params)
    states = m['state']
    N = len(states)

    tmpl = None
    for s in states:
        for t in s.get('trans', []):
            out = t.get('out', '')
            if not out: continue
            tt = parse_tok(out)
            tmpl = tt if tmpl is None else merge_shape(tmpl, tt)
    L = count_leaves(tmpl)

    silent, emitting = [], []
    for s_idx, s in enumerate(states):
        for t in s.get('trans', []):
            w = eval_w(t.get('weight', 1), full)
            if w <= 0: continue
            lw = math.log(w)
            d = t['to']
            out = t.get('out', '')
            if not out:
                silent.append((s_idx, d, lw))
            else:
                profile = []
                decode(parse_tok(out), tmpl, profile)
                deltas = tuple(0 if x is None else 1 for x in profile)
                syms = tuple(profile)
                emitting.append((s_idx, d, deltas, syms, lw))
    silent.sort(key=lambda e: e[1])

    lens = [len(a) for a in leaves]
    total = 1
    for l in lens: total *= (l + 1)
    strides = [1] * L
    for k in range(L - 2, -1, -1):
        strides[k] = strides[k + 1] * (lens[k + 1] + 1)

    f = [-math.inf] * (N * total)
    f[0] = 0.0
    for s_, d_, lw in silent:
        sv = f[s_ * total + 0]
        if sv != -math.inf:
            off = d_ * total + 0
            f[off] = lse(f[off], sv + lw)

    idx = [0] * L
    while True:
        k = L; advanced = False
        while k > 0:
            k -= 1
            if idx[k] < lens[k]:
                idx[k] += 1
                for j in range(k + 1, L): idx[j] = 0
                advanced = True; break
        if not advanced: break
        cell = sum(idx[k] * strides[k] for k in range(L))
        for s_, d_, deltas, syms, lw in emitting:
            ok = True; prev = cell
            for k in range(L):
                if deltas[k]:
                    if idx[k] == 0: ok = False; break
                    if leaves[k][idx[k] - 1] != syms[k]: ok = False; break
                    prev -= strides[k]
            if ok:
                sv = f[s_ * total + prev]
                if sv != -math.inf:
                    off = d_ * total + cell
                    f[off] = lse(f[off], sv + lw)
        for s_, d_, lw in silent:
            sv = f[s_ * total + cell]
            if sv != -math.inf:
                off = d_ * total + cell
                f[off] = lse(f[off], sv + lw)
    return f[(N - 1) * total + (total - 1)]

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
    run([BOSS] + open_args + ['--codegen', crate, '--rust'])
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

    if abs(fwd - ref) > 1e-9: sys.exit("FAIL: Rust Forward != reference")
    if fwd + 1e-12 < vit: sys.exit("FAIL: Forward < Viterbi")
    print("OK")

if __name__ == '__main__':
    main()
