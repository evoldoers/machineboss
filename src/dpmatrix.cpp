#include <iomanip>
#include "dpmatrix.h"
#include "logger.h"

DPMatrix::DPMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair) :
  machine (machine),
  seqPair (seqPair),
  input (machine.inputTokenizer.tokenize (seqPair.input.seq)),
  output (machine.outputTokenizer.tokenize (seqPair.output.seq)),
  inLen (input.size()),
  outLen (output.size()),
  nStates (machine.nStates()),
  env (seqPair)
{
  alloc();
}

DPMatrix::DPMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair, const Envelope& envelope) :
  machine (machine),
  seqPair (seqPair),
  input (machine.inputTokenizer.tokenize (seqPair.input.seq)),
  output (machine.outputTokenizer.tokenize (seqPair.output.seq)),
  inLen (input.size()),
  outLen (output.size()),
  nStates (machine.nStates()),
  env (envelope)
{
  alloc();
}

void DPMatrix::alloc() {
  Assert (env.fits(seqPair), "Envelope/sequence mismatch:\n%s\n%s\n", JsonWriter<Envelope>::toJsonString(env).c_str(), JsonWriter<SeqPair>::toJsonString(seqPair).c_str());
  Assert (env.connected(), "Envelope is not connected:\n%s\n", JsonWriter<Envelope>::toJsonString(env).c_str());
  offsets = env.offsets();  // initializes nCells()
  LogThisAt(7,"Creating matrix with " << nCells() << " cells (<=" << (inLen+1) << "*" << (outLen+1) << "*" << nStates << ")" << endl);
  LogThisAt(8,"Machine:" << endl << machine.toJsonString() << endl);
  cellStorage.resize (nCells(), -numeric_limits<double>::infinity());
}

void DPMatrix::writeJson (ostream& outs) const {
  outs << "{" << endl
       << " \"input\": \"" << seqPair.input.name << "\"," << endl
       << " \"output\": \"" << seqPair.output.name << "\"," << endl
       << " \"cell\": [";
  for (InputIndex i = 0; i <= inLen; ++i)
    for (OutputIndex o = 0; o <= outLen; ++o)
      for (StateIndex s = 0; s < nStates; ++s)
	outs << ((i || o || s) ? "," : "") << endl
	     << "  { \"inPos\": " << i << ", \"outPos\": " << o << ", \"state\": " << machine.state[s].name << ", \"logLike\": " << setprecision(5) << cell(i,o,s) << " }";
  outs << endl
       << " ]" << endl
       << "}" << endl;
}

ostream& operator<< (ostream& out, const DPMatrix& m) {
  m.writeJson (out);
  return out;
}

MachinePath DPMatrix::traceBack (const Machine& m, TransSelector ts) const {
  return traceBack (m, inLen, outLen, nStates - 1, ts);
}

MachinePath DPMatrix::traceBack (const Machine& m, InputIndex inPos, OutputIndex outPos, StateIndex s, TransSelector ts) const {
  MachinePath path;
  TraceTerminator stopTrace = [&] (InputIndex inPos, OutputIndex outPos, StateIndex s, EvaluatedMachineState::TransIndex ti) {
    path.trans.push_front (m.state[s].getTransition (ti));
    return false;
  };
  traceBack (m, inLen, outLen, s, stopTrace, ts);
  return path;
}

void DPMatrix::traceBack (const Machine& m, InputIndex inPos, OutputIndex outPos, StateIndex s, TraceTerminator stopTrace, TransSelector selectTrans) const {
  Assert (cell(inPos,outPos,s) > -numeric_limits<double>::infinity(), "Can't do traceback: no finite-weight paths");
  while (inPos > 0 || outPos > 0 || s != 0) {
    const EvaluatedMachineState& state = machine.state[s];
    double bestLogLike = -numeric_limits<double>::infinity();
    StateIndex bestSource;
    EvaluatedMachineState::TransIndex bestTransIndex;
    TransVisitor tv = selectTrans (inPos, outPos, bestSource, bestTransIndex, bestLogLike);
    const InputToken inTok = inPos ? input[inPos-1] : InputTokenizer::emptyToken();
    const OutputToken outTok = outPos ? output[outPos-1] : OutputTokenizer::emptyToken();
    if (inPos && outPos)
      pathIterate (tv, state.incoming, inTok, outTok, inPos - 1, outPos - 1);
    if (inPos)
      pathIterate (tv, state.incoming, inTok, OutputTokenizer::emptyToken(), inPos - 1, outPos);
    if (outPos)
      pathIterate (tv, state.incoming, InputTokenizer::emptyToken(), outTok, inPos, outPos - 1);
    pathIterate (tv, state.incoming, InputTokenizer::emptyToken(), OutputTokenizer::emptyToken(), inPos, outPos);
    const MachineTransition& bestTrans = m.state[bestSource].getTransition (bestTransIndex);
    if (!bestTrans.inputEmpty()) --inPos;
    if (!bestTrans.outputEmpty()) --outPos;
    s = bestSource;
    if (stopTrace (inPos, outPos, s, bestTransIndex))
      break;
  }
}

MachinePath DPMatrix::traceForward (const Machine& m, TransSelector ts) const {
  return traceBack (m, 0, 0, 0, ts);
}

MachinePath DPMatrix::traceForward (const Machine& m, InputIndex inPos, OutputIndex outPos, StateIndex s, TransSelector ts) const {
  MachinePath path;
  TraceTerminator stopTrace = [&] (InputIndex inPos, OutputIndex outPos, StateIndex s, EvaluatedMachineState::TransIndex ti) {
    path.trans.push_back (m.state[s].getTransition (ti));
    return false;
  };
  traceForward (m, inLen, outLen, s, stopTrace, ts);
  return path;
}

void DPMatrix::traceForward (const Machine& m, InputIndex inPos, OutputIndex outPos, StateIndex s, TraceTerminator stopTrace, TransSelector selectTrans) const {
  Assert (cell(inPos,outPos,s) > -numeric_limits<double>::infinity(), "Can't do traceforward: no finite-weight paths");
  while (inPos < inLen || outPos < outLen || s != nStates - 1) {
    const EvaluatedMachineState& state = machine.state[s];
    double bestLogLike = -numeric_limits<double>::infinity();
    StateIndex bestDest;
    EvaluatedMachineState::TransIndex bestTransIndex;
    TransVisitor tv = selectTrans (inPos, outPos, bestDest, bestTransIndex, bestLogLike);
    const bool endOfInput = (inPos == inLen);
    const bool endOfOutput = (outPos == outLen);
    const InputToken inTok = endOfInput ? InputTokenizer::emptyToken() : input[inPos];
    const OutputToken outTok = endOfOutput ? OutputTokenizer::emptyToken() : output[outPos];
    if (!endOfInput && !endOfOutput)
      pathIterate (tv, state.outgoing, inTok, outTok, inPos + 1, outPos + 1);
    if (!endOfInput)
      pathIterate (tv, state.outgoing, inTok, OutputTokenizer::emptyToken(), inPos + 1, outPos);
    if (!endOfOutput)
      pathIterate (tv, state.outgoing, InputTokenizer::emptyToken(), outTok, inPos, outPos + 1);
    pathIterate (tv, state.outgoing, InputTokenizer::emptyToken(), OutputTokenizer::emptyToken(), inPos, outPos);
    if (stopTrace (inPos, outPos, s, bestTransIndex))
      break;
    const MachineTransition& bestTrans = m.state[s].getTransition (bestTransIndex);
    Assert (bestTrans.dest == bestDest, "Traceforward error");
    if (!bestTrans.inputEmpty()) ++inPos;
    if (!bestTrans.outputEmpty()) ++outPos;
    s = bestDest;
  }
}

DPMatrix::TransVisitor DPMatrix::selectMaxTrans (InputIndex i, OutputIndex o, StateIndex& bestState, EvaluatedMachineState::TransIndex& bestTransIndex, double& bestLogLike) {
  TransVisitor visit = [&] (StateIndex s, EvaluatedMachineState::TransIndex ti, double tll) {
    if (tll > bestLogLike) {
      bestLogLike = tll;
      bestState = s;
      bestTransIndex = ti;
    }
  };
  return visit;
}

DPMatrix::TransSelector DPMatrix::randomTransSelector (mt19937& rng) const {
  uniform_real_distribution<double> distrib (0, 1);
  TransSelector selector = [&] (InputIndex i, OutputIndex o, StateIndex& bestState, EvaluatedMachineState::TransIndex& bestTransIndex, double& bestLogLike) -> TransVisitor {
    TransVisitor visit = [&] (StateIndex s, EvaluatedMachineState::TransIndex ti, double tll) {
      const double cellLogLike = cell (i, o, s);
      const double pKeep = exp(bestLogLike-cellLogLike), pDiscard = exp(tll-cellLogLike);
      const double p = distrib(rng) * (pDiscard + pKeep);
      if (p < pDiscard) {
	bestState = s;
	bestTransIndex = ti;
      }
      log_accum_exp (bestLogLike, tll);
    };
    return visit;
  };
  return selector;
}

