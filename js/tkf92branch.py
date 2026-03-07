#!/usr/bin/env python3
"""Generate TKF92 protein transducer preset JSON for machineboss."""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from machineboss.neural.tkf92 import make_tkf92_machine

m = make_tkf92_machine()

# Build index-to-name map for state name references
idx_to_name = {i: s.name for i, s in enumerate(m.state) if s.name is not None}

# Serialize with state names instead of integer indices
d = {"state": []}
for s in m.state:
    sd = {}
    if s.name is not None:
        sd["id"] = s.name
    trans = []
    for t in s.trans:
        td = {"to": idx_to_name.get(t.dest, t.dest)}
        if t.input:
            td["in"] = t.input
        if t.output:
            td["out"] = t.output
        if t.weight != 1:
            td["weight"] = t.weight
        trans.append(td)
    sd["trans"] = trans
    d["state"].append(sd)

if m.defs:
    d["defs"] = m.defs

d["cons"] = {"rate": ["t", "insRate", "delRate"]}

json.dump(d, sys.stdout)
sys.stdout.write('\n')
