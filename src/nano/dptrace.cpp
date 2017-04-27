#include "dptrace.h"
#include "../logger.h"

TraceDPMatrix::IndexedTrans::IndexedTrans (const EvaluatedMachineState::Trans& t, StateIndex s, StateIndex d, InputToken i)
{
  init (t.logWeight, t.transIndex);
  src = s;
  dest = d;
  in = i;
}

TraceDPMatrix::TraceDPMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const Trace& trace, const TraceParams& traceParams) :
  eval (eval),
  modelParams (modelParams),
  trace (trace),
  traceParams (traceParams),
  moments (trace),
  coeffs (modelParams, traceParams, eval.outputTokenizer),
  outLen (trace.sample.size()),
  nStates (eval.nStates()),
  nOutToks (eval.outputTokenizer.tok2sym.size())
{
  LogThisAt(7,"Creating " << (outLen+1) << "*" << nStates << " matrix" << endl);
  LogThisAt(8,"Machine:" << endl << eval.toJsonString() << endl);

  transByOut.resize (nOutToks);
  for (StateIndex dest = 0; dest < nStates; ++dest)
    for (const auto& inTok_outStateTransMap: eval.state[dest].incoming)
      for (const auto& outTok_stateTransMap: inTok_outStateTransMap.second)
	for (const auto& src_trans: outTok_stateTransMap.second) {
	  Assert (!outTok_stateTransMap.first || (src_trans.first <= dest), "Input-blinded machine is not topologically sorted");
	  transByOut[outTok_stateTransMap.first].push_back (IndexedTrans (src_trans.second, src_trans.first, dest, inTok_outStateTransMap.first));
	}

  cellStorage.resize (nCells(), -numeric_limits<double>::infinity());
}

void TraceDPMatrix::writeJson (ostream& outs) const {
  outs << "{" << endl
       << " \"cell\": [";
  for (OutputIndex o = 0; o <= outLen; ++o)
    for (StateIndex s = 0; s < nStates; ++s)
      outs << ((o || s) ? "," : "") << endl
	   << "  { \"outPos\": " << o
	   << ", \"state\": " << eval.state[s].name
	   << ", \"logLike\": " << setprecision(5) << cell(o,s)
	   << " }";
  outs << endl
       << " ]" << endl
       << "}" << endl;
}

string TraceDPMatrix::toJsonString() const {
  ostringstream outs;
  writeJson (outs);
  return outs.str();
}
