#include "dptrace.h"
#include "../logger.h"

TraceDPMatrix::IndexedTrans::IndexedTrans (const EvaluatedMachineState::Trans& t, StateIndex s, StateIndex d, InputToken i)
{
  init (t.logWeight, t.transIndex);
  src = s;
  dest = d;
  in = i;
}

// S = storageColumns, M = maxStorageColumns, T = totalColumns, X = blockSize
// X + (T/X) = S
// If M is large enough, we can set S=M so that all available storage is used:
//  X + (T/X) = M
//  X^2 - MX + T = 0
//  X = (M + sqrt(M^2 - 4T)) / 2
// The condition for this is M^2 - 4T >= 0
// Otherwise, we just minimize S via dS/dX=0, yielding X=sqrt(T) and S=2*sqrt(T)
size_t calcBlockSize (size_t maxStorageColumns, size_t totalColumns) {
  const double discriminant = pow ((double) maxStorageColumns, 2) - 4 * (double) totalColumns;
  return ceil (discriminant >= 0 ? ((maxStorageColumns + sqrt(discriminant)) / 2) : sqrt ((double) totalColumns));
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
  LogThisAt(8,"Block size is " << blockSize << " columns, # of checkpoint columns is " << nCheckpoints << endl);
  LogThisAt(7,"Creating " << storageColumns << "-column * " << nStates << "-state matrix" << endl);
  columnStorage.resize (storageColumns, vguard<double> (nStates, -numeric_limits<double>::infinity()));
}
