"""Wavefront beam-Viterbi alignment for cyclic transducers.

Uses numpy (not JAX JIT) since the beam is inherently dynamic-shaped.
Organizes DP along anti-diagonal wavefronts d = i + j, where consuming
transitions advance the wavefront and silent transitions are resolved
within each wavefront.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .types import JAXMachine, NEG_INF


class BeamAlignResult(NamedTuple):
    """Result of beam alignment."""
    score: float
    path: list[tuple[int, int, int, int, int, int]]
    # Each path element: (inPos, outPos, srcState, dstState, inTok, outTok)


def beam_align(machine: JAXMachine,
               input_seq: np.ndarray | None,
               output_seq: np.ndarray | None,
               beam_width: int = 100) -> BeamAlignResult:
    """Wavefront beam-Viterbi alignment.

    Works on cyclic machines (e.g. TKF92, Plan7) where standard Viterbi
    requires topological sort.

    Args:
        machine: JAXMachine with dense log_trans tensor.
        input_seq: Array of input tokens (1-indexed, 0=empty). Can be None.
        output_seq: Array of output tokens (1-indexed, 0=empty). Can be None.
        beam_width: Maximum cells to keep per wavefront.

    Returns:
        BeamAlignResult with score and alignment path.
    """
    assert machine.log_trans is not None, "beam_align requires dense log_trans"

    log_trans = np.array(machine.log_trans, dtype=np.float64)
    S = machine.n_states
    Li = len(input_seq) if input_seq is not None else 0
    Lo = len(output_seq) if output_seq is not None else 0
    start_state = 0
    end_state = S - 1

    # Cell: (inPos, outPos, state, score, parentWave, parentIdx, srcState)
    # We store these as parallel numpy arrays per wavefront for efficiency,
    # but use a list-of-dicts approach for clarity and dynamic growth.

    class Beam:
        """A wavefront beam: list of cells with fast key lookup."""
        def __init__(self):
            self.cells = []  # list of [inPos, outPos, state, score, parentWave, parentIdx, srcState]
            self.key_to_idx = {}  # (inPos, outPos, state) -> index

        def add_or_update(self, inPos, outPos, state, score, parentWave, parentIdx, srcState):
            key = (inPos, outPos, state)
            if key in self.key_to_idx:
                idx = self.key_to_idx[key]
                if score > self.cells[idx][3]:
                    self.cells[idx] = [inPos, outPos, state, score, parentWave, parentIdx, srcState]
            else:
                self.key_to_idx[key] = len(self.cells)
                self.cells.append([inPos, outPos, state, score, parentWave, parentIdx, srcState])

        def prune(self, width, wave_idx):
            if len(self.cells) <= width:
                return
            # Sort by score descending, keep top-width
            indices = sorted(range(len(self.cells)), key=lambda i: -self.cells[i][3])
            kept = indices[:width]
            reindex = {}
            new_cells = []
            for new_i, old_i in enumerate(kept):
                reindex[old_i] = new_i
                new_cells.append(self.cells[old_i])
            # Fix up parent indices: remap same-wavefront parents, leave others as-is
            for cell in new_cells:
                if cell[4] == wave_idx:
                    if cell[5] in reindex:
                        cell[5] = reindex[cell[5]]
                    else:
                        # Parent was pruned — invalidate traceback
                        cell[4] = -1
                        cell[5] = -1
            self.cells = new_cells
            self.key_to_idx = {(c[0], c[1], c[2]): i for i, c in enumerate(self.cells)}

        def __len__(self):
            return len(self.cells)

    maxD = Li + Lo
    wavefronts = [None] * (maxD + 1)

    # Build lookup: for each src state, list of (inTok, outTok, dstState, logWeight)
    # Only for transitions with finite weight
    outgoing = [[] for _ in range(S)]
    for src in range(S):
        for in_tok in range(machine.n_input_tokens):
            for out_tok in range(machine.n_output_tokens):
                for dst in range(S):
                    w = float(log_trans[in_tok, out_tok, src, dst])
                    if w > NEG_INF / 2:
                        outgoing[src].append((in_tok, out_tok, dst, w))

    def silent_closure(beam, wave_idx):
        """Propagate silent transitions within a wavefront."""
        changed = True
        while changed:
            changed = False
            n = len(beam.cells)
            for ci in range(n):
                cell = beam.cells[ci]
                srcState = cell[2]
                srcScore = cell[3]
                srcInPos = cell[0]
                srcOutPos = cell[1]
                for in_tok, out_tok, dst, w in outgoing[srcState]:
                    if in_tok == 0 and out_tok == 0:
                        new_score = srcScore + w
                        key = (srcInPos, srcOutPos, dst)
                        if key in beam.key_to_idx:
                            idx = beam.key_to_idx[key]
                            if new_score > beam.cells[idx][3]:
                                beam.cells[idx] = [srcInPos, srcOutPos, dst, new_score,
                                                   wave_idx, ci, srcState]
                                changed = True
                        else:
                            beam.key_to_idx[key] = len(beam.cells)
                            beam.cells.append([srcInPos, srcOutPos, dst, new_score,
                                              wave_idx, ci, srcState])
                            changed = True

    # Initialize wavefront 0
    beam0 = Beam()
    beam0.add_or_update(0, 0, start_state, 0.0, -1, -1, 0)
    silent_closure(beam0, 0)
    beam0.prune(beam_width, 0)
    wavefronts[0] = beam0

    # Fill wavefronts
    for d in range(1, maxD + 1):
        beam = Beam()

        # From d-1: insert (input-only) and delete (output-only)
        if d - 1 >= 0 and wavefronts[d - 1] is not None:
            prev = wavefronts[d - 1]
            for ci, cell in enumerate(prev.cells):
                srcInPos, srcOutPos, srcState, srcScore = cell[0], cell[1], cell[2], cell[3]
                for in_tok, out_tok, dst, w in outgoing[srcState]:
                    # Insert: consumes input only
                    if in_tok != 0 and out_tok == 0:
                        newInPos = srcInPos + 1
                        if newInPos <= Li and input_seq[newInPos - 1] == in_tok:
                            beam.add_or_update(newInPos, srcOutPos, dst,
                                             srcScore + w, d - 1, ci, srcState)
                    # Delete: consumes output only
                    elif in_tok == 0 and out_tok != 0:
                        newOutPos = srcOutPos + 1
                        if newOutPos <= Lo and output_seq[newOutPos - 1] == out_tok:
                            beam.add_or_update(srcInPos, newOutPos, dst,
                                             srcScore + w, d - 1, ci, srcState)

        # From d-2: match (both input and output)
        if d - 2 >= 0 and wavefronts[d - 2] is not None:
            prev = wavefronts[d - 2]
            for ci, cell in enumerate(prev.cells):
                srcInPos, srcOutPos, srcState, srcScore = cell[0], cell[1], cell[2], cell[3]
                for in_tok, out_tok, dst, w in outgoing[srcState]:
                    if in_tok != 0 and out_tok != 0:
                        newInPos = srcInPos + 1
                        newOutPos = srcOutPos + 1
                        if (newInPos <= Li and newOutPos <= Lo
                                and input_seq[newInPos - 1] == in_tok
                                and output_seq[newOutPos - 1] == out_tok):
                            beam.add_or_update(newInPos, newOutPos, dst,
                                             srcScore + w, d - 2, ci, srcState)

        # Silent closure
        silent_closure(beam, d)

        # Prune
        beam.prune(beam_width, d)
        wavefronts[d] = beam

    # Find end cell
    if wavefronts[maxD] is None:
        return BeamAlignResult(score=float('-inf'), path=[])

    end_beam = wavefronts[maxD]
    end_key = (Li, Lo, end_state)
    if end_key not in end_beam.key_to_idx:
        return BeamAlignResult(score=float('-inf'), path=[])

    end_idx = end_beam.key_to_idx[end_key]
    score = end_beam.cells[end_idx][3]

    # Traceback
    path = []
    cur_wave = maxD
    cur_idx = end_idx
    while cur_wave >= 0 and cur_idx >= 0:
        cell = wavefronts[cur_wave].cells[cur_idx]
        parent_wave = cell[4]
        parent_idx = cell[5]
        src_state = cell[6]

        if parent_wave < 0:
            break

        parent_cell = wavefronts[parent_wave].cells[parent_idx]
        di = cell[0] - parent_cell[0]
        dj = cell[1] - parent_cell[1]

        in_tok = input_seq[cell[0] - 1] if di > 0 else 0
        out_tok = output_seq[cell[1] - 1] if dj > 0 else 0

        path.append((cell[0], cell[1], src_state, cell[2], int(in_tok), int(out_tok)))

        cur_wave = parent_wave
        cur_idx = parent_idx

    path.reverse()
    return BeamAlignResult(score=float(score), path=path)
