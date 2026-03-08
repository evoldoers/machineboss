#include "beam_align.h"

using namespace MachineBoss;

BeamAlignMatrix::BeamAlignMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair, size_t beamWidth) :
  machine (machine),
  input (machine.inputTokenizer.tokenize (seqPair.input.seq)),
  output (machine.outputTokenizer.tokenize (seqPair.output.seq)),
  inLen (input.size()),
  outLen (output.size()),
  nStates (machine.nStates()),
  beamWidth (beamWidth)
{
  fill();
}

double BeamAlignMatrix::logLike() const {
  const int lastD = inLen + outLen;
  if (lastD < 0 || lastD >= (int) wavefronts.size())
    return -numeric_limits<double>::infinity();
  const auto& beam = wavefronts[lastD];
  for (const auto& cell : beam)
    if (cell.inPos == inLen && cell.outPos == outLen && cell.state == machine.endState())
      return cell.score;
  return -numeric_limits<double>::infinity();
}

MachinePath BeamAlignMatrix::path (const Machine& m) const {
  // Find end cell
  const int lastD = inLen + outLen;
  int curWave = lastD;
  int curIdx = -1;

  if (curWave < 0 || curWave >= (int) wavefronts.size())
    return MachinePath();

  const auto& lastBeam = wavefronts[curWave];
  for (int i = 0; i < (int) lastBeam.size(); ++i)
    if (lastBeam[i].inPos == inLen && lastBeam[i].outPos == outLen && lastBeam[i].state == machine.endState()) {
      curIdx = i;
      break;
    }

  if (curIdx < 0)
    return MachinePath();

  // Traceback: collect transitions in reverse
  vguard<MachineTransition> revTrans;
  while (curWave >= 0 && curIdx >= 0) {
    const BeamCell& cell = wavefronts[curWave][curIdx];
    if (cell.parentWavefront < 0)
      break;  // reached start

    // Reconstruct the transition from srcState to cell.state
    const StateIndex src = cell.srcState;
    const StateIndex dst = cell.state;

    // Determine what was consumed
    const BeamCell& parentCell = wavefronts[cell.parentWavefront][cell.parentIdx];
    const InputIndex di = cell.inPos - parentCell.inPos;
    const OutputIndex dj = cell.outPos - parentCell.outPos;

    InputSymbol inSym;
    OutputSymbol outSym;
    if (di > 0)
      inSym = machine.inputTokenizer.tok2sym[input[cell.inPos - 1]];
    if (dj > 0)
      outSym = machine.outputTokenizer.tok2sym[output[cell.outPos - 1]];

    // Find the matching transition in the original machine
    const MachineState& srcMachineState = m.state[src];
    for (const auto& trans : srcMachineState.trans) {
      if (trans.dest == dst) {
        bool inMatch = (di > 0) ? (trans.in == inSym) : trans.inputEmpty();
        bool outMatch = (dj > 0) ? (trans.out == outSym) : trans.outputEmpty();
        if (inMatch && outMatch) {
          revTrans.push_back(trans);
          break;
        }
      }
    }

    curWave = cell.parentWavefront;
    curIdx = cell.parentIdx;
  }

  // Reverse to get forward path
  MachinePath result;
  for (auto it = revTrans.rbegin(); it != revTrans.rend(); ++it)
    result.trans.push_back(*it);

  return result;
}

void BeamAlignMatrix::fill() {
  ProgressLog(plogDP,6);
  const int maxD = inLen + outLen;
  wavefronts.resize(maxD + 1);

  // Helper: find or update a cell in a candidate list
  // Uses a map keyed by (inPos, outPos, state) for merging
  typedef tuple<InputIndex, OutputIndex, StateIndex> CellKey;

  auto cellKey = [](const BeamCell& c) -> CellKey {
    return make_tuple(c.inPos, c.outPos, c.state);
  };

  // Initialize wavefront 0: start cell at (0, 0, startState)
  {
    BeamCell startCell;
    startCell.inPos = 0;
    startCell.outPos = 0;
    startCell.state = machine.startState();
    startCell.score = 0;
    startCell.parentWavefront = -1;
    startCell.parentIdx = -1;
    startCell.srcState = 0;

    // Silent closure from start cell
    map<CellKey, int> cellMap;
    vguard<BeamCell> beam;
    beam.push_back(startCell);
    cellMap[cellKey(startCell)] = 0;

    bool changed = true;
    while (changed) {
      changed = false;
      const int beamSize = beam.size();
      for (int ci = 0; ci < beamSize; ++ci) {
        // Copy cell data to avoid dangling reference if push_back reallocates
        const InputIndex srcInPos = beam[ci].inPos;
        const OutputIndex srcOutPos = beam[ci].outPos;
        const StateIndex srcState = beam[ci].state;
        const LogWeight srcScore = beam[ci].score;
        const EvaluatedMachineState& evalState = machine.state[srcState];
        // Silent transitions: inTok=0, outTok=0
        if (evalState.outgoing.count(InputTokenizer::emptyToken())) {
          const auto& outMap = evalState.outgoing.at(InputTokenizer::emptyToken());
          if (outMap.count(OutputTokenizer::emptyToken())) {
            for (const auto& st : outMap.at(OutputTokenizer::emptyToken())) {
              const StateIndex dstState = st.first;
              const LogWeight w = srcScore + st.second.logWeight;
              CellKey key = make_tuple(srcInPos, srcOutPos, dstState);
              if (cellMap.count(key)) {
                int idx = cellMap[key];
                if (w > beam[idx].score) {
                  beam[idx].score = w;
                  beam[idx].parentWavefront = 0;
                  beam[idx].parentIdx = ci;
                  beam[idx].srcState = srcState;
                  changed = true;
                }
              } else {
                BeamCell newCell;
                newCell.inPos = srcInPos;
                newCell.outPos = srcOutPos;
                newCell.state = dstState;
                newCell.score = w;
                newCell.parentWavefront = 0;
                newCell.parentIdx = ci;
                newCell.srcState = srcState;
                cellMap[key] = beam.size();
                beam.push_back(newCell);
                changed = true;
              }
            }
          }
        }
      }
    }

    // Prune
    if (beam.size() > beamWidth) {
      partial_sort(beam.begin(), beam.begin() + beamWidth, beam.end(),
                   [](const BeamCell& a, const BeamCell& b) { return a.score > b.score; });
      beam.resize(beamWidth);
    }
    wavefronts[0] = beam;
  }

  plogDP.initProgress ("Beam-align wavefront DP (%d diagonals)", maxD + 1);

  // Fill wavefronts d=1..maxD
  for (int d = 1; d <= maxD; ++d) {
    plogDP.logProgress (d / (double) (maxD + 1), "diagonal %d/%d", d, maxD);

    map<CellKey, int> cellMap;
    vguard<BeamCell> candidates;

    // Helper to add/merge a candidate cell
    auto addCandidate = [&](InputIndex ip, OutputIndex op, StateIndex dst,
                            LogWeight score, int srcWave, int srcIdx, StateIndex srcSt) {
      CellKey key = make_tuple(ip, op, dst);
      if (cellMap.count(key)) {
        int idx = cellMap[key];
        if (score > candidates[idx].score) {
          candidates[idx].score = score;
          candidates[idx].parentWavefront = srcWave;
          candidates[idx].parentIdx = srcIdx;
          candidates[idx].srcState = srcSt;
        }
      } else {
        BeamCell c;
        c.inPos = ip;
        c.outPos = op;
        c.state = dst;
        c.score = score;
        c.parentWavefront = srcWave;
        c.parentIdx = srcIdx;
        c.srcState = srcSt;
        cellMap[key] = candidates.size();
        candidates.push_back(c);
      }
    };

    // Collect from consuming transitions in previous wavefronts
    // Insert (input-only): from d-1, consumes input[inPos-1]
    if (d >= 1 && d - 1 < (int) wavefronts.size()) {
      const auto& prevBeam = wavefronts[d - 1];
      for (int ci = 0; ci < (int) prevBeam.size(); ++ci) {
        const BeamCell& src = prevBeam[ci];
        const EvaluatedMachineState& evalState = machine.state[src.state];
        const InputIndex newInPos = src.inPos + 1;
        if (newInPos <= inLen) {
          const InputToken inTok = input[newInPos - 1];
          // Insert: consumes input only
          if (evalState.outgoing.count(inTok)) {
            const auto& outMap = evalState.outgoing.at(inTok);
            if (outMap.count(OutputTokenizer::emptyToken())) {
              for (const auto& st : outMap.at(OutputTokenizer::emptyToken())) {
                const LogWeight w = src.score + st.second.logWeight;
                addCandidate(newInPos, src.outPos, st.first, w, d - 1, ci, src.state);
              }
            }
          }
        }
        // Delete: consumes output only
        const OutputIndex newOutPos = src.outPos + 1;
        if (newOutPos <= outLen) {
          const OutputToken outTok = output[newOutPos - 1];
          if (evalState.outgoing.count(InputTokenizer::emptyToken())) {
            const auto& outMap = evalState.outgoing.at(InputTokenizer::emptyToken());
            if (outMap.count(outTok)) {
              for (const auto& st : outMap.at(outTok)) {
                const LogWeight w = src.score + st.second.logWeight;
                addCandidate(src.inPos, newOutPos, st.first, w, d - 1, ci, src.state);
              }
            }
          }
        }
      }
    }

    // Match (input+output): from d-2, consumes both input[inPos-1] and output[outPos-1]
    if (d >= 2 && d - 2 < (int) wavefronts.size()) {
      const auto& prevBeam = wavefronts[d - 2];
      for (int ci = 0; ci < (int) prevBeam.size(); ++ci) {
        const BeamCell& src = prevBeam[ci];
        const EvaluatedMachineState& evalState = machine.state[src.state];
        const InputIndex newInPos = src.inPos + 1;
        const OutputIndex newOutPos = src.outPos + 1;
        if (newInPos <= inLen && newOutPos <= outLen) {
          const InputToken inTok = input[newInPos - 1];
          const OutputToken outTok = output[newOutPos - 1];
          if (evalState.outgoing.count(inTok)) {
            const auto& outMap = evalState.outgoing.at(inTok);
            if (outMap.count(outTok)) {
              for (const auto& st : outMap.at(outTok)) {
                const LogWeight w = src.score + st.second.logWeight;
                addCandidate(newInPos, newOutPos, st.first, w, d - 2, ci, src.state);
              }
            }
          }
        }
      }
    }

    // Silent closure within this wavefront
    // Use index-based access (not references) because push_back can invalidate references
    bool changed = true;
    while (changed) {
      changed = false;
      const int candSize = candidates.size();
      for (int ci = 0; ci < candSize; ++ci) {
        // Copy cell data to avoid dangling reference if push_back reallocates
        const InputIndex srcInPos = candidates[ci].inPos;
        const OutputIndex srcOutPos = candidates[ci].outPos;
        const StateIndex srcState = candidates[ci].state;
        const LogWeight srcScore = candidates[ci].score;
        const EvaluatedMachineState& evalState = machine.state[srcState];
        if (evalState.outgoing.count(InputTokenizer::emptyToken())) {
          const auto& outMap = evalState.outgoing.at(InputTokenizer::emptyToken());
          if (outMap.count(OutputTokenizer::emptyToken())) {
            for (const auto& st : outMap.at(OutputTokenizer::emptyToken())) {
              const StateIndex dstState = st.first;
              const LogWeight w = srcScore + st.second.logWeight;
              CellKey key = make_tuple(srcInPos, srcOutPos, dstState);
              if (cellMap.count(key)) {
                int idx = cellMap[key];
                if (w > candidates[idx].score) {
                  candidates[idx].score = w;
                  candidates[idx].parentWavefront = d;
                  candidates[idx].parentIdx = ci;
                  candidates[idx].srcState = srcState;
                  changed = true;
                }
              } else {
                BeamCell c;
                c.inPos = srcInPos;
                c.outPos = srcOutPos;
                c.state = dstState;
                c.score = w;
                c.parentWavefront = d;
                c.parentIdx = ci;
                c.srcState = srcState;
                cellMap[key] = candidates.size();
                candidates.push_back(c);
                changed = true;
              }
            }
          }
        }
      }
    }

    // Prune to beamWidth
    if (candidates.size() > beamWidth) {
      // Before pruning, we need to remap parentIdx values that point within this wavefront
      // First, sort and keep top-K
      vguard<int> order(candidates.size());
      for (int i = 0; i < (int) order.size(); ++i) order[i] = i;
      partial_sort(order.begin(), order.begin() + beamWidth, order.end(),
                   [&](int a, int b) { return candidates[a].score > candidates[b].score; });

      // Build reindex map for cells within this wavefront
      map<int, int> reindex;
      vguard<BeamCell> pruned(beamWidth);
      for (int i = 0; i < (int) beamWidth; ++i) {
        reindex[order[i]] = i;
        pruned[i] = candidates[order[i]];
      }

      // Fix up parentIdx for cells whose parent is within the same wavefront
      for (auto& cell : pruned) {
        if (cell.parentWavefront == d) {
          if (reindex.count(cell.parentIdx)) {
            cell.parentIdx = reindex[cell.parentIdx];
          } else {
            // Parent was pruned — invalidate traceback
            cell.parentWavefront = -1;
            cell.parentIdx = -1;
          }
        }
      }

      wavefronts[d] = pruned;
    } else {
      wavefronts[d] = candidates;
    }
  }
}
