#include <iomanip>
#include <algorithm>
#include "logger.h"

template<class IndexMapper>
DPMatrix<IndexMapper>::DPMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair) :
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

template<class IndexMapper>
DPMatrix<IndexMapper>::DPMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair, const Envelope& envelope) :
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

template<class IndexMapper>
void DPMatrix<IndexMapper>::alloc() {
  Assert (env.fits(seqPair), "Envelope/sequence mismatch:\n%s\n%s\n", JsonWriter<Envelope>::toJsonString(env).c_str(), JsonWriter<SeqPair>::toJsonString(seqPair).c_str());
  Assert (env.connected(), "Envelope is not connected:\n%s\n", JsonWriter<Envelope>::toJsonString(env).c_str());
  offsets = env.offsets();  // initializes nCells()
  LogThisAt(7,"Creating matrix with " << nCells() << " cells (<=" << (inLen+1) << "*" << (outLen+1) << "*" << nStates << ")" << endl);
  LogThisAt(8,"Machine:" << endl << machine.toJsonString() << endl);
  cellStorage.resize (nCells(), -numeric_limits<double>::infinity());
}

template<class IndexMapper>
void DPMatrix<IndexMapper>::writeJson (ostream& outs) const {
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

template<class IndexMapper>
ostream& operator<< (ostream& out, const DPMatrix<IndexMapper>& m) {
  m.writeJson (out);
  return out;
}

template<class IndexMapper>
MachinePath DPMatrix<IndexMapper>::traceBack (const Machine& m, TransSelector selectTrans) const {
  return traceBack (m, inLen, outLen, nStates - 1, selectTrans);
}

template<class IndexMapper>
MachinePath DPMatrix<IndexMapper>::traceBack (const Machine& m, StateIndex s, TransSelector selectTrans) const {
  return traceBack (m, inLen, outLen, s, selectTrans);
}

template<class IndexMapper>
MachinePath DPMatrix<IndexMapper>::traceBack (const Machine& m, InputIndex inPos, OutputIndex outPos, StateIndex s, TransSelector selectTrans) const {
  MachinePath path;
  TraceTerminator stopTrace = [&] (InputIndex inPos, OutputIndex outPos, StateIndex s, EvaluatedMachineState::TransIndex ti) {
    path.trans.push_front (m.state[s].getTransition (ti));
    return false;
  };
  traceBack (m, inLen, outLen, s, stopTrace, selectTrans);
  return path;
}

template<class IndexMapper>
void DPMatrix<IndexMapper>::traceBack (const Machine& m, InputIndex inPos, OutputIndex outPos, StateIndex s, TraceTerminator stopTrace, TransSelector selectTrans) const {
  Assert (cell(inPos,outPos,s) > -numeric_limits<double>::infinity(), "Can't do traceback: no finite-weight paths");
  while (inPos > 0 || outPos > 0 || s != 0) {
    const EvaluatedMachineState& state = machine.state[s];
    vguard<double> loglike;
    vguard<StateIndex> source;
    vguard<EvaluatedMachineState::TransIndex> transIndex;
    TransVisitor tv = addTransToTraceOptions (source, transIndex, loglike);
    const InputToken inTok = inPos ? input[inPos-1] : InputTokenizer::emptyToken();
    const OutputToken outTok = outPos ? output[outPos-1] : OutputTokenizer::emptyToken();
    if (inPos && outPos)
      pathIterate (tv, state.incoming, inTok, outTok, inPos - 1, outPos - 1);
    if (inPos)
      pathIterate (tv, state.incoming, inTok, OutputTokenizer::emptyToken(), inPos - 1, outPos);
    if (outPos)
      pathIterate (tv, state.incoming, InputTokenizer::emptyToken(), outTok, inPos, outPos - 1);
    pathIterate (tv, state.incoming, InputTokenizer::emptyToken(), OutputTokenizer::emptyToken(), inPos, outPos);
    const size_t best = selectTrans (loglike);
    const auto bestSource = source[best];
    const auto bestTransIndex = transIndex[best];
    const MachineTransition& bestTrans = m.state[bestSource].getTransition (bestTransIndex);
    if (!bestTrans.inputEmpty()) --inPos;
    if (!bestTrans.outputEmpty()) --outPos;
    s = bestSource;
    if (stopTrace (inPos, outPos, s, bestTransIndex))
      break;
  }
}

template<class IndexMapper>
MachinePath DPMatrix<IndexMapper>::traceForward (const Machine& m, TransSelector selectTrans) const {
  return traceBack (m, 0, 0, 0, selectTrans);
}

template<class IndexMapper>
MachinePath DPMatrix<IndexMapper>::traceForward (const Machine& m, InputIndex inPos, OutputIndex outPos, StateIndex s, TransSelector selectTrans) const {
  MachinePath path;
  TraceTerminator stopTrace = [&] (InputIndex inPos, OutputIndex outPos, StateIndex s, EvaluatedMachineState::TransIndex ti) {
    path.trans.push_back (m.state[s].getTransition (ti));
    return false;
  };
  traceForward (m, inLen, outLen, s, stopTrace, selectTrans);
  return path;
}

template<class IndexMapper>
void DPMatrix<IndexMapper>::traceForward (const Machine& m, InputIndex inPos, OutputIndex outPos, StateIndex s, TraceTerminator stopTrace, TransSelector selectTrans) const {
  Assert (cell(inPos,outPos,s) > -numeric_limits<double>::infinity(), "Can't do traceforward: no finite-weight paths");
  while (inPos < inLen || outPos < outLen || s != nStates - 1) {
    const EvaluatedMachineState& state = machine.state[s];
    vguard<double> loglike;
    vguard<StateIndex> dest;
    vguard<EvaluatedMachineState::TransIndex> transIndex;
    TransVisitor tv = addTransToTraceOptions (dest, transIndex, loglike);
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
    const size_t best = selectTrans (loglike);
    const auto bestDest = dest[best];
    const auto bestTransIndex = transIndex[best];
    if (stopTrace (inPos, outPos, s, bestTransIndex))
      break;
    const MachineTransition& bestTrans = m.state[s].getTransition (bestTransIndex);
    Assert (bestTrans.dest == bestDest, "Traceforward error");
    if (!bestTrans.inputEmpty()) ++inPos;
    if (!bestTrans.outputEmpty()) ++outPos;
    s = bestDest;
  }
}

template<class IndexMapper>
typename DPMatrix<IndexMapper>::TransVisitor DPMatrix<IndexMapper>::addTransToTraceOptions (vguard<StateIndex>& state, vguard<EvaluatedMachineState::TransIndex>& transIndex, vguard<double>& loglike) {
  TransVisitor visit = [&] (StateIndex s, EvaluatedMachineState::TransIndex ti, double tll) {
    state.push_back (s);
    transIndex.push_back (ti);
    loglike.push_back (tll);
  };
  return visit;
}

template<class IndexMapper>
size_t DPMatrix<IndexMapper>::selectMaxTrans (const vguard<double>& logWeights) {
  return distance (logWeights.begin(), max_element (logWeights.begin(), logWeights.end()));
}

template<class IndexMapper>
typename DPMatrix<IndexMapper>::TransSelector DPMatrix<IndexMapper>::randomTransSelector (mt19937& rng) {
  TransSelector selector = [&] (const vguard<double>& logWeights) -> size_t {
    vguard<double> weights;
    weights.reserve (logWeights.size());
    for (const auto lw: logWeights)
      weights.push_back (exp (lw));
    return random_index (weights, rng);
  };
  return selector;
}

