#include "dptrace.h"
#include "../logger.h"

TraceDPMatrix::IndexedTrans::IndexedTrans (const EvaluatedMachineState::Trans& t, StateIndex s, StateIndex d, InputToken i)
{
  init (t.logWeight, t.transIndex);
  src = s;
  dest = d;
  in = i;
}

size_t calcBlockSize (size_t storageColumns, size_t totalColumns) {
  const double s = pow ((double) storageColumns, 2) - 4*totalColumns;
  return s >= 0 ? ceil ((storageColumns + sqrt(s)) / 2) : 1;
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
  blockSize (blockBytes
	     ? max ((OutputIndex) 2,
		    min ((OutputIndex) nColumns,
			 (OutputIndex) calcBlockSize (blockBytes / (nStates * sizeof(double)), outLen)))
	     : nColumns),
  nCheckpoints (1 + ((nColumns - 1) / blockSize))
{
  LogThisAt(7,"Indexing " << nStates << "-state machine" << endl);
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

  const OutputIndex storageColumns = blockSize + nCheckpoints - 1;
  LogThisAt(7,"Creating " << storageColumns << "-column (" << (outLen+1) << "-sample) * " << nStates << "-state matrix" << endl);
  columnStorage.resize (storageColumns, vguard<double> (nStates, -numeric_limits<double>::infinity()));
}
