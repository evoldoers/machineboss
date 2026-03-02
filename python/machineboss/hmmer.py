"""Pure Python HMMER3 file parser.

Mirrors the C++ HmmerModel class, producing Machine objects.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TextIO

from .machine import Machine, MachineState, MachineTransition

# SwissProt background amino acid frequencies
SWISSPROT_BG: dict[str, float] = {
    "A": 0.0825, "C": 0.0138, "D": 0.0546, "E": 0.0673,
    "F": 0.0386, "G": 0.0708, "H": 0.0227, "I": 0.0592,
    "K": 0.0581, "L": 0.0965, "M": 0.0241, "N": 0.0405,
    "P": 0.0473, "Q": 0.0393, "R": 0.0553, "S": 0.0663,
    "T": 0.0535, "V": 0.0686, "W": 0.0109, "Y": 0.0292,
}


@dataclass
class HmmerNode:
    match_emit: list[float] = field(default_factory=list)
    ins_emit: list[float] = field(default_factory=list)
    m_to_m: float = 0.0
    m_to_i: float = 0.0
    m_to_d: float = 0.0
    i_to_m: float = 0.0
    i_to_i: float = 0.0
    d_to_m: float = 0.0
    d_to_d: float = 0.0


@dataclass
class HmmerModel:
    """HMMER3 profile HMM model."""
    alph: list[str] = field(default_factory=list)
    nodes: list[HmmerNode] = field(default_factory=list)
    ins0_emit: list[float] = field(default_factory=list)
    null_emit: list[float] = field(default_factory=list)
    b_to_m1: float = 0.0
    b_to_i0: float = 0.0
    b_to_d1: float = 0.0
    i0_to_m1: float = 0.0
    i0_to_i0: float = 0.0

    @staticmethod
    def str_to_prob(s: str) -> float:
        return 0.0 if s == "*" else math.exp(-float(s))

    @classmethod
    def read(cls, f: TextIO) -> HmmerModel:
        """Parse an HMMER3 format file."""
        model = cls()
        tag_re = re.compile(r"^([A-Z]+)")
        end_re = re.compile(r"^//")

        for line in f:
            m = tag_re.match(line)
            if m and m.group(1) == "HMM":
                tokens = line.split()
                assert len(tokens) > 1, "HMM parse error: empty alphabet"
                model.alph = tokens[1:]

                # Skip transition header line, COMPO line, node 0 insert emission line
                for _ in range(3):
                    line = next(f)

                # Node 0 insert emissions
                ins0 = line.split()
                assert len(ins0) == len(model.alph)
                model.ins0_emit = [cls.str_to_prob(s) for s in ins0]

                # Begin transitions
                line = next(f)
                bt = line.split()
                model.b_to_m1 = cls.str_to_prob(bt[0])
                model.b_to_i0 = cls.str_to_prob(bt[1])
                model.b_to_d1 = cls.str_to_prob(bt[2])
                model.i0_to_m1 = cls.str_to_prob(bt[3])
                model.i0_to_i0 = cls.str_to_prob(bt[4])

                # Parse nodes
                for line in f:
                    if end_re.match(line):
                        break
                    match_fields = line.split()
                    assert len(match_fields) == len(model.alph) + 6

                    line = next(f)  # insert emission line
                    ins_fields = line.split()

                    line = next(f)  # transition line
                    trans_fields = line.split()
                    assert len(trans_fields) == 7

                    node = HmmerNode(
                        match_emit=[cls.str_to_prob(s) for s in match_fields[1:len(model.alph)+1]],
                        ins_emit=[cls.str_to_prob(s) for s in ins_fields],
                        m_to_m=cls.str_to_prob(trans_fields[0]),
                        m_to_i=cls.str_to_prob(trans_fields[1]),
                        m_to_d=cls.str_to_prob(trans_fields[2]),
                        i_to_m=cls.str_to_prob(trans_fields[3]),
                        i_to_i=cls.str_to_prob(trans_fields[4]),
                        d_to_m=cls.str_to_prob(trans_fields[5]),
                        d_to_d=cls.str_to_prob(trans_fields[6]),
                    )
                    model.nodes.append(node)
                break

        model._load_null_model()
        return model

    def _load_null_model(self) -> None:
        self.null_emit = []
        for sym in self.alph:
            self.null_emit.append(SWISSPROT_BG.get(sym, 1.0 / len(self.alph)))

    # State index methods (same formulas as C++)
    def b_idx(self) -> int:
        return 0

    def ix_idx(self, n: int) -> int:
        return 5 * n + 1

    def i_idx(self, n: int) -> int:
        return 5 * n + 2

    def mx_idx(self, n: int) -> int:
        return 5 * n - 2

    def m_idx(self, n: int) -> int:
        return 5 * n - 1

    def d_idx(self, n: int) -> int:
        return 5 * n

    def core_end_idx(self) -> int:
        return 5 * len(self.nodes) + 3

    def n_core_states(self) -> int:
        return 5 * len(self.nodes) + 4

    def calc_match_occupancy(self) -> list[float]:
        M = len(self.nodes)
        mocc = [0.0] * M
        mocc[1] = self.nodes[0].m_to_i + self.nodes[0].m_to_m
        for k in range(2, M):
            mocc[k] = (mocc[k-1] * (self.nodes[k].m_to_m + self.nodes[k].m_to_i)
                        + (1.0 - mocc[k-1]) * self.nodes[k].d_to_m)
        return mocc

    def machine(self, local: bool = True) -> Machine:
        """Build core-only machine (same as C++ HmmerModel::machine)."""
        M = len(self.nodes)
        assert M > 0

        states = [MachineState() for _ in range(self.n_core_states())]

        # B state
        states[self.b_idx()].name = "B"
        if local:
            occ = self.calc_match_occupancy()
            Z = sum(occ[k] * (M - k + 1) for k in range(1, M))
            for k in range(1, M):
                states[self.b_idx()].trans.append(
                    MachineTransition(dest=self.m_idx(k), weight=occ[k] / Z))
        else:
            states[self.b_idx()].trans.append(
                MachineTransition(dest=self.m_idx(1), weight=self.b_to_m1))
            states[self.b_idx()].trans.append(
                MachineTransition(dest=self.i_idx(0), weight=self.b_to_i0))
            states[self.b_idx()].trans.append(
                MachineTransition(dest=self.d_idx(1), weight=self.b_to_d1))

        # I0, Ix0
        states[self.ix_idx(0)].trans.append(
            MachineTransition(dest=self.m_idx(1), weight=self.i0_to_m1))
        states[self.ix_idx(0)].trans.append(
            MachineTransition(dest=self.i_idx(0), weight=self.i0_to_i0))

        for sym_i, sym in enumerate(self.alph):
            states[self.i_idx(0)].trans.append(
                MachineTransition(dest=self.ix_idx(0), output=sym,
                                  weight=self.ins0_emit[sym_i]))

        for n in range(M + 1):
            ns = str(n)
            states[self.i_idx(n)].name = f"I{ns}"
            states[self.ix_idx(n)].name = f"Ix{ns}"

            if n > 0:
                nd = self.nodes[n - 1]
                states[self.m_idx(n)].name = f"M{ns}"
                states[self.mx_idx(n)].name = f"Mx{ns}"
                states[self.d_idx(n)].name = f"D{ns}"

                end = (n == M)

                if end:
                    if not local:
                        states[self.mx_idx(n)].trans.append(
                            MachineTransition(dest=self.core_end_idx(), weight=nd.m_to_m))
                else:
                    states[self.mx_idx(n)].trans.append(
                        MachineTransition(dest=self.m_idx(n + 1), weight=nd.m_to_m))
                states[self.mx_idx(n)].trans.append(
                    MachineTransition(dest=self.i_idx(n), weight=nd.m_to_i))
                if not end:
                    states[self.mx_idx(n)].trans.append(
                        MachineTransition(dest=self.d_idx(n + 1), weight=nd.m_to_d))

                dest_after_i = self.core_end_idx() if end else self.m_idx(n + 1)
                states[self.ix_idx(n)].trans.append(
                    MachineTransition(dest=dest_after_i, weight=nd.i_to_m))
                states[self.ix_idx(n)].trans.append(
                    MachineTransition(dest=self.i_idx(n), weight=nd.i_to_i))

                if end:
                    if not local:
                        states[self.d_idx(n)].trans.append(
                            MachineTransition(dest=self.core_end_idx(), weight=nd.d_to_m))
                else:
                    states[self.d_idx(n)].trans.append(
                        MachineTransition(dest=self.m_idx(n + 1), weight=nd.d_to_m))
                    states[self.d_idx(n)].trans.append(
                        MachineTransition(dest=self.d_idx(n + 1), weight=nd.d_to_d))

                for sym_i, sym in enumerate(self.alph):
                    states[self.m_idx(n)].trans.append(
                        MachineTransition(dest=self.mx_idx(n), output=sym,
                                          weight=nd.match_emit[sym_i]))
                    states[self.i_idx(n)].trans.append(
                        MachineTransition(dest=self.ix_idx(n), output=sym,
                                          weight=nd.ins_emit[sym_i]))

                if local:
                    states[self.m_idx(n)].trans.append(
                        MachineTransition(dest=self.core_end_idx(), weight=1))
                    states[self.d_idx(n)].trans.append(
                        MachineTransition(dest=self.core_end_idx(), weight=1))

        states[self.core_end_idx()].name = "E"
        return Machine(state=states)

    def plan7_machine(self, multihit: bool = False, L: float = 400) -> Machine:
        """Build full Plan7 machine with N/C/J/T flanking states."""
        assert len(self.nodes) > 0
        assert len(self.null_emit) == len(self.alph)

        n_core = self.n_core_states()
        # Plan7 state indices (appended after core)
        n_idx = n_core
        nx_idx = n_core + 1
        p7_b_idx = n_core + 2
        cx_idx = n_core + 3
        c_idx = n_core + 4
        jx_idx = n_core + 5
        j_idx = n_core + 6
        t_idx = n_core + 7
        n_total = n_core + 8

        # Build core in local mode
        core = self.machine(local=True)

        states = [MachineState() for _ in range(n_total)]
        for i in range(n_core):
            states[i] = core.state[i]

        # Move B's transitions to Plan7 Begin state
        states[p7_b_idx] = states[self.b_idx()]
        states[p7_b_idx].name = "B"

        # Repurpose index 0 as S
        states[self.b_idx()] = MachineState(name="S", trans=[
            MachineTransition(dest=nx_idx, weight=1.0)
        ])

        # N-terminal flank
        states[n_idx] = MachineState(name="N", trans=[
            MachineTransition(dest=nx_idx, output=sym, weight=self.null_emit[i])
            for i, sym in enumerate(self.alph)
        ])
        states[nx_idx] = MachineState(name="Nx", trans=[
            MachineTransition(dest=n_idx, weight=L / (L + 1)),
            MachineTransition(dest=p7_b_idx, weight=1.0 / (L + 1)),
        ])

        # E transitions
        if multihit:
            states[self.core_end_idx()].trans.extend([
                MachineTransition(dest=cx_idx, weight=0.5),
                MachineTransition(dest=jx_idx, weight=0.5),
            ])
        else:
            states[self.core_end_idx()].trans.append(
                MachineTransition(dest=cx_idx, weight=1.0))

        # C-terminal flank
        states[c_idx] = MachineState(name="C", trans=[
            MachineTransition(dest=cx_idx, output=sym, weight=self.null_emit[i])
            for i, sym in enumerate(self.alph)
        ])
        states[cx_idx] = MachineState(name="Cx", trans=[
            MachineTransition(dest=c_idx, weight=L / (L + 1)),
            MachineTransition(dest=t_idx, weight=1.0 / (L + 1)),
        ])

        # J loop
        if multihit:
            states[j_idx] = MachineState(name="J", trans=[
                MachineTransition(dest=jx_idx, output=sym, weight=self.null_emit[i])
                for i, sym in enumerate(self.alph)
            ])
            states[jx_idx] = MachineState(name="Jx", trans=[
                MachineTransition(dest=j_idx, weight=L / (L + 1)),
                MachineTransition(dest=p7_b_idx, weight=1.0 / (L + 1)),
            ])
        else:
            states[j_idx] = MachineState(name="J")
            states[jx_idx] = MachineState(name="Jx")

        # Terminal
        states[t_idx] = MachineState(name="T")

        return Machine(state=states)
