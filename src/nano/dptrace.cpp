#include "dptrace.h"

TraceDPMatrix::TraceDPMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const Trace& trace, const TraceParams& traceParams) :
  eval (eval),
  modelParams (modelParams),
  trace (trace),
  traceParams (traceParams),
  moments (trace),
  coeffs (modelParams, traceParams, eval.outputTokenizer),
  outLen (trace.sample.size()),
  nStates (eval.nStates())
{
  LogThisAt(7,"Creating " << (outLen+1) << "*" << nStates << " matrix" << endl);
  LogThisAt(8,"Machine:" << endl << eval.toJsonString() << endl);
  cellStorage.resize (nCells());
  logEmit.resize (coeffs.gauss.size());
}

void TraceDPMatrix::writeJson (ostream& outs) const {
  outs << "{" << endl
       << " \"cell\": [";
  for (OutputIndex o = 0; o <= outLen; ++o)
    for (StateIndex s = 0; s < nStates; ++s)
      outs << ((o || s) ? "," : "") << endl
	   << "  { \"outPos\": " << o
	   << ", \"state\": " << machine.state[s].name
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
