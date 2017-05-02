#include "dptrace.h"
#include "../logger.h"

TraceDPMatrix::IndexedTrans::IndexedTrans (const EvaluatedMachineState::Trans& t, StateIndex s, StateIndex d, InputToken i)
{
  init (t.logWeight, t.transIndex);
  src = s;
  dest = d;
  in = i;
}

TraceDPMatrix::TraceDPMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const TraceMoments& moments, const TraceParams& traceParams, size_t bb) :
  eval (eval),
  modelParams (modelParams),
  moments (moments),
  traceParams (traceParams),
  coeffs (modelParams, traceParams, eval.outputTokenizer),
  outLen (moments.sample.size()),
  nStates (eval.nStates()),
  nOutToks (eval.outputTokenizer.tok2sym.size()),
  nColumns (outLen + 1),
  blockBytes (bb),
  blockSize (blockBytes ? max((OutputIndex)2,min((OutputIndex)(blockBytes / (nStates * sizeof(double))),(OutputIndex)nColumns)) : nColumns),
  nCheckpoints (1 + ((nColumns - 1) / blockSize))
{
  LogThisAt(7,"Creating " << (outLen+1) << "-sample * " << nStates << "-state matrix" << endl);
  LogThisAt(9,"Machine:" << endl << eval.toJsonString() << endl);

  nTrans = 0;
  transByOut.resize (nOutToks);
  for (StateIndex dest = 0; dest < nStates; ++dest)
    for (const auto& inTok_outStateTransMap: eval.state[dest].incoming)
      for (const auto& outTok_stateTransMap: inTok_outStateTransMap.second)
	for (const auto& src_trans: outTok_stateTransMap.second) {
	  Assert (outTok_stateTransMap.first || (src_trans.first <= dest), "Input-blinded machine is not topologically sorted (transition from %d to %d has no output)", src_trans.first, dest);
	  transByOut[outTok_stateTransMap.first].push_back (IndexedTrans (src_trans.second, src_trans.first, dest, inTok_outStateTransMap.first));
	  ++nTrans;
	}

  columnStorage.resize (blockSize + nCheckpoints - 1, vguard<double> (nStates, -numeric_limits<double>::infinity()));
}
