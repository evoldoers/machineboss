#!/usr/bin/env python3
"""Forward / Backward / forward_backward_counts / machine.json regression.

Codegens an echo phylo, builds the crate, and exercises:
  - forward == backward to floating-point noise
  - forward_backward_counts.log_likelihood matches forward
  - forward_backward_counts returns NUM_BUCKETS counts, all >= 0
  - sum of state posteriors at the origin cell ≈ 1
  - sum of state posteriors at the last cell ≈ 1
  - to_machine_json produces parseable JSON with no remaining __C
    placeholders, and every transition has a numeric `expected_count`
    field.
"""

import os, sys, subprocess, json, tempfile

REPO = os.environ.get('REPO_ROOT', os.getcwd())
BOSS = os.path.join(REPO, 'bin', 'boss')

def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        sys.stderr.write(f"FAIL: {cmd}\n{r.stderr}\n")
        sys.exit(1)
    return r.stdout

def main():
    workdir = tempfile.mkdtemp(prefix='fb_')
    crate = os.path.join(workdir, 'crate')
    echo = os.path.join(REPO, 't', 'machine', 'echo-with-time.json')

    run([BOSS, '--pair-json', '--generate-uniform', '01', '-m',
         '--begin', echo, '--phylo-tree-string', '(A,B)P;', '--end',
         '--codegen', crate, '--rust'])

    # Confirm machine.json was emitted and has __C sentinels.
    with open(os.path.join(crate, 'machine.json')) as f:
        mj = f.read()
    if '__C' not in mj:
        print('FAIL: machine.json should contain __C placeholders'); sys.exit(1)
    # It should NOT be parseable as JSON at this stage (sentinels are not JSON).
    try:
        json.loads(mj); print('FAIL: machine.json template should not parse as JSON'); sys.exit(1)
    except json.JSONDecodeError:
        pass

    check_rs = os.path.join(crate, 'examples', 'check.rs')
    os.makedirs(os.path.dirname(check_rs), exist_ok=True)
    with open(check_rs, 'w') as f:
        f.write('''use phylo_dp::{
    forward, backward, forward_backward_counts,
    forward_matrix, backward_matrix, state_log_posterior,
    Params, ALPHABET, NUM_LEAVES, NUM_STATES, NUM_BUCKETS,
};
fn idx(c: char) -> u32 { ALPHABET.iter().position(|x| x.chars().next() == Some(c)).unwrap() as u32 }
fn main() {
    let p = Params { t_A_: 0.4, t_B_: 0.7 };
    let a: Vec<u32> = "01".chars().map(idx).collect();
    let b: Vec<u32> = "01".chars().map(idx).collect();
    let f = forward(&p, [&a, &b]);
    let bw = backward(&p, [&a, &b]);
    let fb = forward_backward_counts(&p, [&a, &b]);
    let fm = forward_matrix(&p, [&a, &b]);
    let bm = backward_matrix(&p, [&a, &b]);

    // Marginalize state posteriors at the origin cell and the final cell.
    let origin = [0usize; NUM_LEAVES];
    let mut full = [0usize; NUM_LEAVES];
    for k in 0..NUM_LEAVES { full[k] = [a.len(), b.len()][k]; }
    let mut sum_origin = 0.0;
    let mut sum_final = 0.0;
    for s in 0..NUM_STATES {
        let lp_o = state_log_posterior(&fm, &bm, s as u32, origin);
        if lp_o.is_finite() { sum_origin += lp_o.exp(); }
        let lp_f = state_log_posterior(&fm, &bm, s as u32, full);
        if lp_f.is_finite() { sum_final += lp_f.exp(); }
    }

    let mut json_out = String::new();
    fb.to_machine_json(&mut json_out);

    println!("forward       = {}", f);
    println!("backward      = {}", bw);
    println!("fb.log_lik    = {}", fb.log_likelihood);
    println!("counts.len    = {}", fb.expected_counts.len());
    println!("NUM_BUCKETS   = {}", NUM_BUCKETS);
    println!("sum_state_origin = {}", sum_origin);
    println!("sum_state_final  = {}", sum_final);
    println!("counts_negative_count = {}", fb.expected_counts.iter().filter(|&&c| c < 0.0).count());
    println!("json_len = {}", json_out.len());
    println!("json_has_placeholder = {}", json_out.contains("__C"));
}
''')
    run(['cargo', 'build', '--release', '--example', 'check', '--quiet'], cwd=crate)
    out = run(['cargo', 'run', '--release', '--example', 'check', '--quiet'], cwd=crate)

    vals = {}
    for line in out.strip().splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            vals[k.strip()] = v.strip()

    fwd = float(vals['forward'])
    bwd = float(vals['backward'])
    fbll = float(vals['fb.log_lik'])
    nbuckets = int(vals['NUM_BUCKETS'])
    counts_len = int(vals['counts.len'])
    negs = int(vals['counts_negative_count'])
    sum_o = float(vals['sum_state_origin'])
    sum_f = float(vals['sum_state_final'])
    json_has = vals['json_has_placeholder']

    print(out, end='')
    fail = False
    if abs(fwd - bwd) > 1e-12: print(f"FAIL: |fwd-bwd|={abs(fwd-bwd):.3e}"); fail = True
    if abs(fwd - fbll) > 1e-12: print(f"FAIL: |fwd-fb.log_lik|={abs(fwd-fbll):.3e}"); fail = True
    if counts_len != nbuckets: print(f"FAIL: counts.len {counts_len} != NUM_BUCKETS {nbuckets}"); fail = True
    if negs > 0: print(f"FAIL: {negs} negative expected_counts"); fail = True
    # Forward-Backward state posterior at origin and full should each sum to 1
    # (start state always reached at origin; end state always reached at full).
    if abs(sum_o - 1.0) > 1e-9: print(f"FAIL: sum_state_origin {sum_o} != 1"); fail = True
    if abs(sum_f - 1.0) > 1e-9: print(f"FAIL: sum_state_final {sum_f} != 1"); fail = True
    if json_has != 'false': print(f"FAIL: machine.json still has placeholders"); fail = True

    # Confirm the rendered JSON is parseable and has expected_count fields.
    rendered = run(['cargo', 'run', '--release', '--example', 'check', '--quiet'], cwd=crate)
    # Re-render using the API directly through a small helper binary that prints just JSON.
    helper = os.path.join(crate, 'examples', 'render.rs')
    with open(helper, 'w') as f:
        f.write('''use phylo_dp::{forward_backward_counts, Params, ALPHABET};
fn idx(c: char) -> u32 { ALPHABET.iter().position(|x| x.chars().next() == Some(c)).unwrap() as u32 }
fn main() {
    let p = Params { t_A_: 0.4, t_B_: 0.7 };
    let a: Vec<u32> = "01".chars().map(idx).collect();
    let b: Vec<u32> = "01".chars().map(idx).collect();
    let fb = forward_backward_counts(&p, [&a, &b]);
    let mut s = String::new(); fb.to_machine_json(&mut s);
    println!("{}", s);
}
''')
    rendered_json = run(['cargo', 'run', '--release', '--example', 'render', '--quiet'], cwd=crate)
    parsed = json.loads(rendered_json)
    n_with_count = 0
    for s in parsed['state']:
        for t in s.get('trans', []):
            if 'expected_count' not in t:
                print(f"FAIL: trans missing expected_count: {t}"); fail = True
            else:
                n_with_count += 1
    print(f"json transitions with expected_count: {n_with_count}")
    if fail: sys.exit(1)
    print("OK")

if __name__ == '__main__':
    main()
