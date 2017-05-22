#ifndef DPTRACE_INCLUDED
#define DPTRACE_INCLUDED

#include "../eval.h"
#include "../logsumexp.h"
#include "moments.h"

class TraceDPMatrix {
public:
  typedef long OutputIndex;

  struct IndexedTrans : EvaluatedMachineState::Trans {
    StateIndex src, dest;
    InputToken in;
    OutputToken out;
    bool loop;
    EvaluatedMachineState::TransIndex loopTransIndex;
    LogWeight loopLogWeight;
    IndexedTrans (const EvaluatedMachineState::Trans&, StateIndex, StateIndex, InputToken, OutputToken);
  };

private:
  typedef long long CellIndex;
  typedef long long EmitIndex;

  vguard<vguard<double> > columnStorage;

  inline size_t columnIndex (OutputIndex outPos) const {
    const long blockOffset = outPos % blockSize;
    return blockOffset ? (blockOffset + nCheckpoints - 1) : (outPos / blockSize);
  }

protected:
  vguard<IndexedTrans> emitTrans, nullTrans;
  size_t nTrans, maxDistanceFromStart;
  vguard<size_t> emitTransOffset;   // if idx = emitTransOffset[d], then emitTrans[idx] is first emit transition to a state of distance >= d from start
  
  void initColumn (vguard<double>& col) {
    fill (col.begin(), col.end(), -numeric_limits<double>::infinity());
  }

  inline OutputIndex checkpoint (OutputIndex outPos) const {
    return outPos - (outPos % blockSize);
  }

  inline double fracBandCenter (OutputIndex outPos) const {
    return max (halfBandWidth, min (1. - halfBandWidth, outPos / (double) outLen));
  }
  
  inline vguard<IndexedTrans>::const_iterator bandTransBegin (OutputIndex outPos) const {
    const size_t d = (size_t) (maxDistanceFromStart * max (0., fracBandCenter(outPos) - halfBandWidth));
    return emitTrans.begin() + emitTransOffset[d];
  }

  inline vguard<IndexedTrans>::const_iterator bandTransEnd (OutputIndex outPos) const {
    const size_t d = (size_t) (maxDistanceFromStart * min (1., fracBandCenter(outPos) + halfBandWidth));
    return emitTrans.begin() + emitTransOffset[d+1];
  }

public:
  const EvaluatedMachine& eval;
  const GaussianModelParams& modelParams;
  const TraceMoments& moments;
  const TraceParams& traceParams;
  const GaussianModelCoefficients coeffs;
  const OutputIndex outLen;
  const StateIndex nStates;
  const OutputToken nOutToks;
  const size_t nColumns;
  const size_t blockBytes;
  const OutputIndex blockSize;
  const size_t nCheckpoints;
  const double bandWidth;
  const double halfBandWidth;
  
  TraceDPMatrix (const EvaluatedMachine&, const GaussianModelParams&, const TraceMoments&, const TraceParams&, size_t blockBytes = 0, double bandWidth = 1);

  inline vguard<double>& column (OutputIndex outPos) { return columnStorage[columnIndex(outPos)]; }
  inline const vguard<double>& column (OutputIndex outPos) const { return columnStorage[columnIndex(outPos)]; }
  
  inline double& cell (OutputIndex outPos, StateIndex state) { return column(outPos)[state]; }
  inline const double cell (OutputIndex outPos, StateIndex state) const { return column(outPos)[state]; }

  inline double logEmitProb (OutputIndex outPos, OutputToken outTok) const {
    return coeffs.gauss[outTok-1].logEmitProb (moments.sample[outPos-1]);
  }

  inline const EvaluatedMachineState::Trans* getLoopTrans (InputToken inTok, OutputToken outTok, StateIndex state) const {
    return (eval.state[state].outgoing.count(inTok)
	    && eval.state[state].outgoing.at(inTok).count(outTok)
	    && eval.state[state].outgoing.at(inTok).at(outTok).count(state))
      ? &eval.state[state].outgoing.at(inTok).at(outTok).at(state)
      : NULL;
  }
  
  inline double logTransProb (OutputIndex outPos, const IndexedTrans& trans) const {
    return logTransProb (outPos, trans.logWeight, trans.loopLogWeight);
  }

  inline double logTransProb (OutputIndex outPos, double logWeight, double loopLogWeight) const {
    const auto& mom = moments.sample[outPos-1];
    return logWeight + (mom.m0 == 1 ? 0. : ((mom.m0 - 1) * loopLogWeight));
  }

  inline double logIncomingProb (const InputToken inTok, const OutputToken outTok, const OutputIndex outPos, const StateIndex src, const StateIndex dest, const EvaluatedMachineState::Trans& trans) const {
    if (outTok) {
      const EvaluatedMachineState::Trans* loopTrans = getLoopTrans(inTok,outTok,dest);
      return cell(outPos-1,src)
	+ logTransProb(outPos,trans.logWeight,loopTrans ? loopTrans->logWeight : -numeric_limits<double>::infinity())
	+ logEmitProb(outPos,outTok);
    } else
      return cell(outPos,src) + trans.logWeight;
  }

  void writeJson (ostream&);
  friend ostream& operator<< (ostream&, TraceDPMatrix&);
};

#endif /* DPTRACE_INCLUDED */
