"""Shared reference implementation for the Rust codegen verification tests.

Implements the same multidimensional Forward DP that the codegen emits,
in plain Python with exact log_sum_exp via math.log1p(math.exp(.)).

Used by check_*.py tests; not standalone runnable on its own.
"""
import json, math


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


def expand_defs(defs, params):
    """Iteratively resolve all `defs` to numeric values given free `params`."""
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
    """Parse a pair-token. Mirrors the codegen's heuristic: try JSON when
    the leading char looks JSON-like, otherwise treat as a literal symbol
    (the L=1 case where no intersection has occurred)."""
    if not tok: return ""
    c = tok[0]
    if c in '[\"-' or c.isdigit():
        try: return json.loads(tok)
        except json.JSONDecodeError: pass
    return tok


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
    """Run multidim Forward (exact lse) over a phylo-composed machine.

    `machine_json` is the JSON string emitted by `boss --pair-json ...`.
    `leaves` is a list of L sequences, each a list of symbol strings."""
    m = json.loads(machine_json) if isinstance(machine_json, str) else machine_json
    full = expand_defs(m.get('defs', {}), params)
    states = m['state']
    N = len(states)

    tmpl = None
    for s in states:
        for t in s.get('trans', []):
            out = t.get('out', '')
            if not out: continue
            tt = parse_tok(out)
            tmpl = tt if tmpl is None else merge_shape(tmpl, tt)
    if tmpl is None:
        raise ValueError("machine has no emitting transitions")
    L = count_leaves(tmpl)
    assert L == len(leaves), f"L={L} vs {len(leaves)} leaves"

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
