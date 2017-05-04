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
    IndexedTrans (const EvaluatedMachineState::Trans&, StateIndex, StateIndex, InputToken);
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
  vguard<vguard<IndexedTrans> > transByOut;  // indexed by output token
  inline const vguard<IndexedTrans>& nullTrans() const { return transByOut.front(); }
  size_t nTrans;

  void initColumn (vguard<double>& col) {
    fill (col.begin(), col.end(), -numeric_limits<double>::infinity());
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
  
  TraceDPMatrix (const EvaluatedMachine&, const GaussianModelParams&, const TraceMoments&, const TraceParams&, size_t blockBytes = 0);

  inline vguard<double>& column (OutputIndex outPos) { return columnStorage[columnIndex(outPos)]; }
  inline const vguard<double>& column (OutputIndex outPos) const { return columnStorage[columnIndex(outPos)]; }
  
  inline double& cell (OutputIndex outPos, StateIndex state) { return column(outPos)[state]; }
  inline const double cell (OutputIndex outPos, StateIndex state) const { return column(outPos)[state]; }

  inline double logEmitProb (OutputIndex outPos, OutputToken outTok) const {
    return coeffs.gauss[outTok-1].logEmitProb (moments.sample[outPos-1]);
  }
};

#endif /* DPTRACE_INCLUDED */
