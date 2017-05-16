#include "dptrace.h"
#include "../logger.h"

TraceDPMatrix::IndexedTrans::IndexedTrans (const EvaluatedMachineState::Trans& t, StateIndex s, StateIndex d, InputToken i, OutputToken o) :
  loopLogWeight (-numeric_limits<double>::infinity()),
  loopTransIndex (0),
  loop (false)
{
  init (t.logWeight, t.transIndex);
  src = s;
  dest = d;
  in = i;
  out = o;
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

TraceDPMatrix::TraceDPMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const TraceMoments& moments, const TraceParams& traceParams, size_t bb, double bandWidth) :
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
  nCheckpoints (1 + ((nColumns - 1) / blockSize)),
  bandWidth (bandWidth)
{
  LogThisAt(7,"Indexing " << nStates << "-state machine" << endl);
  LogThisAt(9,"Machine:" << endl << eval.toJsonString() << endl);

  // use depth-first search to find shortest emit path from start state to each state
  vguard<OutputIndex> minDistanceFromStart (nStates, nStates);
  list<StateIndex> toVisit;
  toVisit.push_back(0);
  minDistanceFromStart[0] = 0;
  while (!toVisit.empty()) {
    const StateIndex src = toVisit.front();
    toVisit.pop_front();
    const OutputIndex len = minDistanceFromStart[src];
    for (const auto& inTok_outStateTransMap: eval.state[src].outgoing)
      for (const auto& outTok_stateTransMap: inTok_outStateTransMap.second) {
	const OutputIndex nextLen = outTok_stateTransMap.first ? (len + 1) : len;
	for (const auto& dest_trans: outTok_stateTransMap.second) {
	  const StateIndex dest = dest_trans.first;
	  if (minDistanceFromStart[dest] > nextLen) {
	    minDistanceFromStart[dest] = nextLen;
	    toVisit.push_back(dest);
	  }
	}
      }
  }
  maxDistanceFromStart = *(max_element (minDistanceFromStart.begin(), minDistanceFromStart.end()));
  
  nTrans = 0;
  size_t nEmitTrans = 0;
  list<IndexedTrans> nullTransList;
  vguard<list<IndexedTrans> > emitTransListByDistance (maxDistanceFromStart + 1);
  for (StateIndex dest = 0; dest < nStates; ++dest)
    for (const auto& inTok_outStateTransMap: eval.state[dest].incoming)
      for (const auto& outTok_stateTransMap: inTok_outStateTransMap.second)
	for (const auto& src_trans: outTok_stateTransMap.second) {
	  Assert (outTok_stateTransMap.first || (src_trans.first <= dest), "Input-blinded machine is not topologically sorted (transition from %d to %d has no output)", src_trans.first, dest);
	  const InputToken inTok = inTok_outStateTransMap.first;
	  const OutputToken outTok = outTok_stateTransMap.first;
	  const StateIndex src = src_trans.first;
	  const EvaluatedMachineState::Trans& trans = src_trans.second;
	  IndexedTrans it (trans, src, dest, inTok, outTok);
	  const EvaluatedMachineState::Trans* loopTrans = getLoopTrans(inTok,outTok,dest);
	  if (loopTrans) {
	    it.loop = true;
	    it.loopTransIndex = loopTrans->transIndex;
	    it.loopLogWeight = loopTrans->logWeight;
	  }
	  (outTok ? emitTransListByDistance[minDistanceFromStart[dest]] : nullTransList).push_back (it);
	  ++nTrans;
	  if (outTok)
	    ++nEmitTrans;
	}
  nullTrans = vguard<IndexedTrans> (nullTransList.begin(), nullTransList.end());
  emitTrans.reserve (nEmitTrans);
  emitTransOffset.resize (maxDistanceFromStart + 2, nEmitTrans);
  for (size_t distance = 0; distance <= maxDistanceFromStart; ++distance) {
    emitTransOffset[distance] = emitTrans.size();
    emitTrans.insert (emitTrans.end(), emitTransListByDistance[distance].begin(), emitTransListByDistance[distance].end());
  }
  
  const OutputIndex storageColumns = blockSize + nCheckpoints - 1;
  LogThisAt(8,"Block size is " << blockSize << " columns, # of checkpoint columns is " << nCheckpoints << endl);
  LogThisAt(7,"Creating " << storageColumns << "-column * " << nStates << "-state matrix" << endl);
  columnStorage.resize (storageColumns, vguard<double> (nStates, -numeric_limits<double>::infinity()));
}

void TraceDPMatrix::writeJson (ostream& out) {
  out << "{" << endl
       << " \"cell\": [";
  for (OutputIndex o = 0; o <= outLen; ++o)
    for (StateIndex s = 0; s < nStates; ++s)
      out << ((o || s) ? "," : "") << endl
	  << "  { \"outPos\": " << o << ", \"state\": " << s << ", \"logLike\": " << setprecision(5) << cell(o,s) << " }";
  out << endl
      << " ]" << endl
      << "}" << endl;
}

ostream& operator<< (ostream& out, TraceDPMatrix& m) {
  m.writeJson (out);
  return out;
}
