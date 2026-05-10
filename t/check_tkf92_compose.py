#!/usr/bin/env python3
"""Cross-check that the two factorisations of the TKF92 joint pair HMM
produce the same marginal Forward log-likelihood:

  L_a = log P(Y) computed via compose(--tkf92-root, --tkf92-branch [6-state])
  L_b = log P(Y) computed via compose(--evolmoves-root, --evolmoves-branch [5-state])

For any output sequence Y, both should be equal (both are valid
generator factorisations of the same joint pair HMM, just with the
length prior split between the singlet and the conditional WFST in
different ways). If they agree to ~1e-9, both factorisations are
internally consistent.

We additionally smoke-test the empty-output case: log P(Y=∅) marginalises
the joint over input X, and the analytical closed form is

    log P(Y=∅) = log[ (1−β)(1−κ) +
                      Σ_{n≥1} (1−β)·κ·(1−α) · (r + (1−r)(1−γ)κ(1−α))^{n−1}
                                            · (1−r)(1−γ)(1−κ) ]
              = log[ (1−β)(1−κ) +
                     (1−β)·κ·(1−α)·(1−r)(1−γ)(1−κ) /
                     (1 − (r + (1−r)(1−γ)κ(1−α))) ]

so we verify the two factorisations both match this closed form.
"""
import os, sys, subprocess, json, tempfile, math

REPO = os.environ.get('REPO_ROOT', os.getcwd())
BOSS = os.path.join(REPO, 'bin', 'boss')

def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        sys.stderr.write(f"FAIL: {cmd}\n{r.stderr}\n{r.stdout}\n")
        sys.exit(1)
    return r.stdout

def compose_loglike(root_args, branch_args, params, output_chars):
    """Run compose(root, branch) via boss and compute Forward log-prob of
    the given output_chars (empty string = length-0 marginal)."""
    root_json   = run([BOSS] + root_args + ['--strip-names'])
    branch_json = run([BOSS] + branch_args + ['--strip-names'])
    rf = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    bf = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    pf = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    rf.write(root_json);   rf.close()
    bf.write(branch_json); bf.close()
    json.dump(params, pf); pf.close()
    try:
        out = run([BOSS, rf.name, bf.name, '-P', pf.name,
                   '--output-chars', output_chars, '-L'])
    finally:
        os.unlink(rf.name); os.unlink(bf.name); os.unlink(pf.name)
    return json.loads(out)[0][2]

def closed_form_logp_empty(p):
    """Analytical log P(Y=∅) marginalising joint over input X. Sums:
       no-input path           : (1−β)(1−κ)
       any-length-input via D  : (1−β)·κ·(1−α) · D-self-loop-sum · (1−r)(1−γ)(1−κ)"""
    a, b, g = p['alpha'], p['beta'], p['gamma']
    k, r    = p['kappa'], p['r']
    one_b   = 1 - b; one_g = 1 - g
    one_k   = 1 - k; one_r = 1 - r; one_a = 1 - a
    no_input = one_b * one_k
    d_loop   = r + one_r * one_g * k * one_a
    if d_loop >= 1.0:
        # Improper distribution; analytical sum diverges. Skip.
        return None
    with_input = (one_b * k * one_a) * (one_r * one_g * one_k) / (1 - d_loop)
    return math.log(no_input + with_input)

def main():
    insRate, delRate, t, r = 0.1, 0.12, 0.5, 0.5
    # Compute auxiliary structural quantities (α, β, γ, κ).
    alpha = math.exp(-delRate * t)
    kappa = insRate / delRate
    delIns = math.exp(-(delRate - insRate) * t)
    beta   = (insRate * (1 - delIns)) / (delRate - insRate * delIns)
    gamma  = 1 - (delRate / insRate) * (beta / (1 - alpha))
    structural = {'alpha': alpha, 'beta': beta, 'gamma': gamma,
                  'kappa': kappa, 'r': r}
    print(f"  α={alpha:.6f}  β={beta:.6f}  γ={gamma:.6f}  κ={kappa:.6f}  r={r}")

    params = {'insRate': insRate, 'delRate': delRate,
              'time': t, 't': t, 'r': r, 'flipRate': 0.1}

    # Closed form for Y = empty.
    cf_empty = closed_form_logp_empty(structural)
    if cf_empty is None:
        print("  (skip closed-form check; D-loop ≥ 1)")

    failures = 0

    # Compare factorisations on a few output sequences.
    test_outputs = ['', '0', '01', '011', '0101']
    print()
    print(f"{'output':>8s} {'tkf92':>16s} {'evolmoves':>16s} {'closed-form':>16s}  diff")
    for y in test_outputs:
        try:
            la = compose_loglike(['--tkf92-root-binary-bsc'],
                                 ['--tkf92-branch-binary-bsc'],
                                 params, y)
        except SystemExit:
            la = float('nan')
        try:
            lb = compose_loglike(['--evolmoves-root-binary-bsc'],
                                 ['--evolmoves-branch-binary-bsc'],
                                 params, y)
        except SystemExit:
            lb = float('nan')
        cf_str = ''
        if y == '' and cf_empty is not None:
            cf_str = f'{cf_empty:>16.10f}'
        diff = abs(la - lb) if math.isfinite(la) and math.isfinite(lb) else float('inf')
        ok = diff < 1e-9
        print(f"{y!r:>8s} {la:>16.10f} {lb:>16.10f} {cf_str:>16s}  {diff:.2e}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1
        # Also verify against closed form (only for empty Y).
        if y == '' and cf_empty is not None:
            for label, val in (('tkf92', la), ('evolmoves', lb)):
                cf_diff = abs(val - cf_empty)
                if cf_diff > 1e-9:
                    print(f"    {label} vs closed-form diff = {cf_diff:.3e}  FAIL")
                    failures += 1

    if failures:
        print(f"\nFAIL ({failures} mismatches)")
        sys.exit(1)
    print("\nOK")

if __name__ == '__main__':
    main()
