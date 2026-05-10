#!/usr/bin/env python3
"""Validate the skeleton → full symbol-expansion design.

Given the M_skel produced by `--phylo-skeleton`, the original branch transducer
T, and a tree, the script walks M_skel's state names (now ordered arrays
following the compose/intersect bug fix in machine.cpp) to recover the
per-branch T-state at every state, looks up T's symbol-aware transitions for
each (src, dst) state pair on each branch, and cross-multiplies under the
intersect input-sync constraint to produce a fully-expanded phylo machine.
The expansion is then fed back through boss for a Forward log-likelihood
calculation that should match (bit-exactly) the legacy `--phylo-no-felsenstein`
path's output on the same observation.

Status: bit-exact match on trees with NO INTERNAL NODES (binary trees with
two leaves directly under the root, or polytomies with all leaves directly
under the root). Trees with internal nodes (e.g. the protein quartet
`((A,B)P,(C,D)Q)R;`) require an additional Felsenstein-per-column sum over
internal-node symbols to reproduce the legacy weight expressions. That
generalization is a follow-up; this script reports whether each test case
matches and exits non-zero if any in-scope case differs.
"""
import os, sys, subprocess, json, tempfile, copy, re

REPO = os.environ.get('REPO_ROOT', '/Users/yam/machineboss')
BOSS = os.path.join(REPO, 'bin', 'boss')

def boss(args, stdin=None):
    r = subprocess.run([BOSS] + args, capture_output=True, text=True, input=stdin)
    if r.returncode != 0:
        sys.stderr.write(f"FAIL {args}\n{r.stderr[:500]}\n")
        sys.exit(1)
    return r.stdout

# ------------------------- Newick parser (lightweight) -------------------------

class TreeNode:
    __slots__ = ('name', 'children', 'parent', 'branch_length')
    def __init__(self, name='', branch_length=None):
        self.name = name
        self.children = []
        self.parent = None
        self.branch_length = branch_length

def parse_newick(s):
    """Return root TreeNode."""
    pos = [0]
    def skip_ws():
        while pos[0] < len(s) and s[pos[0]].isspace(): pos[0] += 1
    def parse_subtree():
        skip_ws()
        node = TreeNode()
        if pos[0] < len(s) and s[pos[0]] == '(':
            pos[0] += 1  # consume '('
            while True:
                child = parse_subtree()
                child.parent = node
                node.children.append(child)
                skip_ws()
                if pos[0] < len(s) and s[pos[0]] == ',':
                    pos[0] += 1; continue
                if pos[0] < len(s) and s[pos[0]] == ')':
                    pos[0] += 1; break
                raise ValueError(f'parse error at {pos[0]}')
        # name
        skip_ws()
        m = re.match(r"[^(),:;\s]*", s[pos[0]:])
        if m:
            node.name = m.group(0)
            pos[0] += len(node.name)
        # branch length
        skip_ws()
        if pos[0] < len(s) and s[pos[0]] == ':':
            pos[0] += 1
            m = re.match(r"[+-]?[0-9.eE+-]+", s[pos[0]:])
            if m:
                node.branch_length = float(m.group(0))
                pos[0] += len(m.group(0))
        return node
    root = parse_subtree()
    skip_ws()
    if pos[0] < len(s) and s[pos[0]] == ';':
        pos[0] += 1
    return root

def list_leaves(root):
    out = []
    def walk(n):
        if not n.children: out.append(n)
        else:
            for c in n.children: walk(c)
    walk(root)
    return out

def list_branches(root):
    """All non-root nodes (each represents a branch from its parent)."""
    out = []
    def walk(n):
        if n.parent is not None: out.append(n)
        for c in n.children: walk(c)
    walk(root)
    return out

# ------------------------- Renaming -------------------------

def rename_expr(e, sub):
    if isinstance(e, str): return sub.get(e, e)
    if isinstance(e, dict): return {k: rename_expr(v, sub) for k, v in e.items()}
    if isinstance(e, list): return [rename_expr(x, sub) for x in e]
    return e

def rename_for_branch(T, time_param, node_name):
    suffix = f"[{node_name}]"
    sub = {time_param: time_param + suffix}
    for k in T.get('defs', {}): sub[k] = k + suffix
    out = copy.deepcopy(T)
    for s in out['state']:
        for t in s.get('trans', []):
            if 'weight' in t: t['weight'] = rename_expr(t['weight'], sub)
    if 'defs' in out:
        renamed_defs = {}
        for k, v in out['defs'].items():
            renamed_defs[k + suffix] = rename_expr(v, sub)
        out['defs'] = renamed_defs
    if 'cons' in out:
        cons = out['cons']
        for key in ('rate','prob'):
            if key in cons:
                cons[key] = [(p+suffix if p == time_param else p) for p in cons[key]]
        if 'norm' in cons:
            cons['norm'] = [[(p+suffix if p == time_param else p) for p in g] for g in cons['norm']]
    return out

# ------------------------- T-branch indexing -------------------------

def index_T(T):
    name_to_idx = {}
    for i, s in enumerate(T['state']):
        nm = s.get('id')
        if isinstance(nm, str): name_to_idx[nm] = i
    pairs = {}
    for i, s in enumerate(T['state']):
        for t in s.get('trans', []):
            pairs.setdefault((i, t['to']), []).append(t)
    return pairs, name_to_idx

# ------------------------- M_skel state-name decoding -------------------------

def unwrap_wait(state_name):
    """waitingMachine wraps a state's name as {"wait": original_name}.
    Recursively unwrap. Returns original name."""
    while isinstance(state_name, dict) and len(state_name) == 1 and 'wait' in state_name:
        state_name = state_name['wait']
    return state_name

def decode_recursive(state_name, node):
    """Walk a M_skel state name following the buildSubtree recursion at `node`.
    Returns dict {branch_name: T_state_name} for every non-root node in the subtree.
    """
    state_name = unwrap_wait(state_name)
    if not node.children:
        # leaf: state_name is the alphabet array (skeleton: ['*']); no T-state info
        return {}
    if len(node.children) == 1:
        # compose: state_name = [T_branch_state, sub_state]
        c = node.children[0]
        T_state, sub_state = state_name[0], state_name[1]
        out = {c.name: T_state}
        out.update(decode_recursive(sub_state, c))
        return out
    # n >= 2 children: left-fold intersect
    return decode_fold(state_name, node.children)

def decode_fold(state_name, children):
    """state_name encodes left-fold intersect of len(children) branches."""
    state_name = unwrap_wait(state_name)
    n = len(children)
    if n == 2:
        out = {}
        for c, bs in zip(children, [state_name[0], state_name[1]]):
            bs = unwrap_wait(bs)
            T_state, sub_state = bs[0], bs[1]
            out[c.name] = T_state
            out.update(decode_recursive(sub_state, c))
        return out
    # n > 2: state_name = [acc, last_branch_compose_state]
    acc, last_branch = state_name[0], state_name[1]
    last_branch = unwrap_wait(last_branch)
    last = children[-1]
    T_state, sub_state = last_branch[0], last_branch[1]
    out = {last.name: T_state}
    out.update(decode_recursive(sub_state, last))
    out.update(decode_fold(acc, children[:-1]))
    return out

# ------------------------- Pair-token encoder -------------------------

PAIR_SEP, PAIR_OPEN, PAIR_CLOSE, PAIR_ESCAPE = ',', '[', ']', '\\'

def needs_wrap(s):
    return any(c in s for c in (PAIR_SEP, PAIR_OPEN, PAIR_CLOSE, PAIR_ESCAPE))

def wrap(s):
    out = PAIR_OPEN
    for c in s:
        if c in (PAIR_OPEN, PAIR_CLOSE, PAIR_ESCAPE):
            out += PAIR_ESCAPE
        out += c
    out += PAIR_CLOSE
    return out

def encode_pair(a, b):
    """Mirror Machine::encodePairToken in default mode."""
    if a == '' and b == '': return ''
    a_ = wrap(a) if needs_wrap(a) else a
    b_ = wrap(b) if needs_wrap(b) else b
    return a_ + PAIR_SEP + b_

def encode_per_leaf_to_pair_token(per_leaf, node):
    """Given per-leaf outputs (dict by leaf name) and the tree, build the
    nested pair-token that intersect/compose would produce at `node`."""
    if not node.children:
        return per_leaf.get(node.name, '')
    # degree 1: compose returns child's pair-token (the branch only adds outputs
    # at the branch transition, not nested-pair structure)
    if len(node.children) == 1:
        return encode_per_leaf_to_pair_token(per_leaf, node.children[0])
    # degree >= 2: fold-left intersect produces nested pair-tokens.
    # acc starts as branch_0's pair-token; each step encode_pair(acc, branch_i).
    parts = [encode_per_leaf_to_pair_token(per_leaf, c) for c in node.children]
    acc = parts[0]
    for p in parts[1:]:
        acc = encode_pair(acc, p)
    return acc

# ------------------------- Weight algebra helpers -------------------------

def is_one(w):
    return w == 1 or w == 1.0

def mul(w1, w2):
    if is_one(w1): return w2
    if is_one(w2): return w1
    return {'*': [w1, w2]}

# ------------------------- Expansion -------------------------

def expand_skeleton(M_skel, T, root, time_param):
    branches = list_branches(root)        # ordered list of TreeNode (non-root)
    leaves = list_leaves(root)
    branch_names = [b.name for b in branches]

    # Per-branch T (renamed)
    Tb = {b.name: rename_for_branch(T, time_param, b.name) for b in branches}
    indexed = {}
    n2i = {}
    for nm, tb in Tb.items():
        pairs, mp = index_T(tb)
        indexed[nm] = pairs
        n2i[nm] = mp

    # Decode each M_skel state into per-branch T state names
    per_state_branch_name = []
    for s in M_skel['state']:
        nm = s.get('id')
        if nm is None:
            per_state_branch_name.append(None)
            continue
        per_state_branch_name.append(decode_recursive(nm, root))

    new_state = []
    for si, s in enumerate(M_skel['state']):
        new_s = {'n': si}
        if 'id' in s: new_s['id'] = s['id']
        new_trans = []

        src_branch = per_state_branch_name[si]

        for t in s.get('trans', []):
            dst = t['to']
            dst_branch = per_state_branch_name[dst]
            if src_branch is None or dst_branch is None:
                new_trans.append(t)
                continue

            # Silent transitions (in='' AND out='') in M_skel preserve their
            # full structural weight (skeletonisation only resets EMIT weights
            # to 1; silent chain-collapsing in ergodicMachine produces direct
            # silent edges with the correct compounded weight). Just copy.
            if t.get('in', '') == '' and t.get('out', '') == '':
                new_trans.append(dict(t))
                continue

            # For each branch, gather candidate transitions and frozen-status
            cand = {}
            frozen = {}
            for bn in branch_names:
                src_name = src_branch[bn]
                dst_name = dst_branch[bn]
                src_idx = n2i[bn][src_name]
                dst_idx = n2i[bn][dst_name]
                cand[bn] = indexed[bn].get((src_idx, dst_idx), [])
                frozen[bn] = (src_name == dst_name)

            # Cross-product of (branch transition or stay) under intersect rules.
            # Algorithmic approach: enumerate "advance set" (the set of branches
            # that fire a non-stay transition for this M_skel transition). The
            # rest stay frozen. For each advance-set choice consistent with the
            # frozen mask (advance-set = non-frozen branches), enumerate the
            # cross-product of cand[b] for b in advance_set, subject to the
            # input-sync rule.
            #
            # In our case the advance set IS exactly the non-frozen branches:
            # they all advance per this M_skel transition. So we cross-product
            # cand[b] for non-frozen b, filter by input-sync, and produce the
            # joint transition.
            advance = [bn for bn in branch_names if not frozen[bn]]

            if not advance:
                continue  # no-op

            # Iterate cross-product
            from itertools import product
            for combo in product(*[cand[bn] for bn in advance]):
                # Input-sync rule: when intersect synchronizes input, all
                # advancing branches with non-empty input must agree.
                inputs = [(adv_bn, t.get('in', '')) for adv_bn, t in zip(advance, combo)]
                # We need: either ALL non-empty inputs are equal, or there's
                # a unique non-empty input that one branch consumes while
                # others have empty input (silent advance for those).
                # In intersect semantics: a transition in the result has
                # input = ONE common input. If some branch has empty input,
                # it's "silent advance" and doesn't constrain.
                # All branches with NON-empty input must agree.
                non_empty_ins = set(i for _, i in inputs if i != '')
                if len(non_empty_ins) > 1:
                    continue
                in_sym = list(non_empty_ins)[0] if non_empty_ins else ''

                # Per-leaf output: for each leaf, walk up to find which
                # branch's transition (if any) emitted to this leaf's
                # subtree. A match/insert at branch B emits a child-symbol
                # which propagates through B's subtree to leaves.
                #
                # Concretely: leaf L emits in this column iff the path from
                # root to L has at least one branch that 'emitted into the
                # leaf' — that's encoded via the wildEcho composition. The
                # leaf-side of the pair-token equals the symbol emitted by
                # the branch closest to the leaf that produced output.
                #
                # For binary tree (A,B)P;: leaves={A,B}. Branches BA, BB.
                # If BA's transition emits 'X' as output, and BA's wildEcho
                # echoes it through to leaf A's slot, then leaf A's output
                # is 'X'. Same for B.
                # For deeper trees: same principle applies — each leaf's
                # output is the output of its IMMEDIATE-PARENT-BRANCH's
                # transition, IF that branch is in advance set with non-
                # empty output. Otherwise empty.

                per_leaf_out = {}
                advance_dict = {bn: t for bn, t in zip(advance, combo)}
                for leaf in leaves:
                    parent_branch_name = leaf.name
                    if parent_branch_name in advance_dict:
                        per_leaf_out[leaf.name] = advance_dict[parent_branch_name].get('out', '')
                    else:
                        per_leaf_out[leaf.name] = ''

                # ALSO: internal-branch transitions can emit a child symbol
                # (e.g. BP match emits a symbol that propagates DOWN through
                # subtree-P). The downstream branches BA/BB then propagate
                # it further. For a column where BP matches (emitting Y),
                # BA matches Y (consuming Y from BP's output) and emits
                # leaf-A symbol; same for BB.
                # IMPORTANT: in intersect, internal branches' OUTPUT becomes
                # an INPUT for the nested subtree. So BP's output is consumed
                # BY the subtree's intersect. The subtree's leaves emit their
                # own symbols based on what BP fed them.
                # → The emit direction flows DOWN: each leaf's output is its
                # nearest-branch's output, regardless of upstream. The
                # upstream's output is consumed by intermediate composes.

                # Pair-token encoded recursively over the tree
                out_pair = encode_per_leaf_to_pair_token(per_leaf_out, root)

                weight = 1
                for adv_bn, tt in zip(advance, combo):
                    weight = mul(weight, tt.get('weight', 1))

                trans_out = {'to': dst}
                if in_sym: trans_out['in'] = in_sym
                if out_pair: trans_out['out'] = out_pair
                trans_out['weight'] = weight
                new_trans.append(trans_out)

        new_s['trans'] = new_trans
        new_state.append(new_s)

    # merge defs / cons
    out = {'state': new_state, 'defs': {}, 'cons': {'rate': [], 'prob': [], 'norm': []}}
    seen_rate, seen_prob = set(), set()
    for tb in Tb.values():
        for k, v in tb.get('defs', {}).items():
            out['defs'][k] = v
        cons = tb.get('cons', {})
        for r in cons.get('rate', []):
            if r not in seen_rate: out['cons']['rate'].append(r); seen_rate.add(r)
        for p in cons.get('prob', []):
            if p not in seen_prob: out['cons']['prob'].append(p); seen_prob.add(p)
        for g in cons.get('norm', []):
            if g not in out['cons']['norm']: out['cons']['norm'].append(g)
    return out

# ------------------------- Test driver -------------------------

def compute_loglike(workdir, machine_path, parent_seq, leaf_cols, params, parent_path=None, cols_path=None, params_path=None):
    if parent_path is None:
        parent_path = os.path.join(workdir, 'parent.json')
        json.dump({'sequence': parent_seq}, open(parent_path, 'w'))
    if cols_path is None:
        cols_path = os.path.join(workdir, 'cols.json')
        json.dump({'sequence': leaf_cols}, open(cols_path, 'w'))
    if params_path is None:
        params_path = os.path.join(workdir, 'params.json')
        json.dump(params, open(params_path, 'w'))
    out = boss(['--generate-json', parent_path, '-m',
                machine_path,
                '--recognize-json', cols_path,
                '-P', params_path, '-L'])
    return json.loads(out)[0][2]

def run_case(label, tree_str, leaf_cols, parent_seq, params, time_param='time'):
    work = tempfile.mkdtemp(prefix=f'exp4_{label}_')
    print(f"\n=== {label}: {tree_str} ===")
    T = json.loads(boss(['--tkf91-branch-dna-jc']))
    M_skel = json.loads(boss(['--tkf91-branch-dna-jc',
                              '--phylo-tree-string', tree_str,
                              '--phylo-time-param', time_param,
                              '--phylo-no-felsenstein',
                              '--phylo-skeleton']))
    M_full = json.loads(boss(['--tkf91-branch-dna-jc',
                              '--phylo-tree-string', tree_str,
                              '--phylo-time-param', time_param,
                              '--phylo-no-felsenstein']))
    print(f"  M_skel: {len(M_skel['state']):>4} states, {sum(len(s.get('trans',[])) for s in M_skel['state']):>6} trans")
    print(f"  M_full: {len(M_full['state']):>4} states, {sum(len(s.get('trans',[])) for s in M_full['state']):>6} trans")

    root = parse_newick(tree_str)
    M_exp = expand_skeleton(M_skel, T, root, time_param)
    print(f"  M_exp:  {len(M_exp['state']):>4} states, {sum(len(s.get('trans',[])) for s in M_exp['state']):>6} trans")

    full_path = os.path.join(work, 'full.json')
    exp_path = os.path.join(work, 'exp.json')
    json.dump(M_full, open(full_path, 'w'))
    json.dump(M_exp,  open(exp_path,  'w'))

    ll_full = compute_loglike(work, full_path, parent_seq, leaf_cols, params)
    ll_exp  = compute_loglike(work, exp_path,  parent_seq, leaf_cols, params)
    eq = abs(ll_full - ll_exp) < 1e-9
    print(f"  -L full={ll_full:.10f}  exp={ll_exp:.10f}  {'MATCH' if eq else 'DIFFER'}")
    return eq

def main():
    ok = True
    # Binary tree with two leaves at root: no internal nodes.
    ok &= run_case('binary',
        '(A,B)P;',
        ['A,A', 'C,C'],
        ['A','C'],
        {'insRate':0.005, 'delRate':0.01, 'time[A]':0.3, 'time[B]':0.2})
    # Polytomy: three leaves at root. No internal nodes.
    ok &= run_case('polytomy3',
        '(A,B,C)P;',
        ['[A,A],A','[C,C],C'],
        ['A','C'],
        {'insRate':0.005, 'delRate':0.01, 'time[A]':0.3, 'time[B]':0.2, 'time[C]':0.4})
    # Trees with internal nodes (e.g. the quartet ((A,B)P,(C,D)Q)R;) currently
    # over-generate per-symbol transitions because the expansion does not yet
    # include the Felsenstein sum over internal-node symbol assignments. Those
    # cases are intentionally not in this validation set; see top-of-file.
    print("\n" + ("ALL OK" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
