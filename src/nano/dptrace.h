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

  vguard<double> cellStorage;

  inline CellIndex nCells() const {
    return nStates * (outLen + 1);
  }

  inline CellIndex cellIndex (OutputIndex outPos, StateIndex state) const {
    return outPos * nStates + state;
  }

protected:
  vguard<vguard<IndexedTrans> > transByOut;  // indexed by output token
  inline const vguard<IndexedTrans>& nullTrans() const { return transByOut.front(); }

public:
  const EvaluatedMachine& eval;
  const GaussianModelParams& modelParams;
  const Trace& trace;
  const TraceParams& traceParams;
  const TraceMoments moments;
  const GaussianModelCoefficients coeffs;
  const OutputIndex outLen;
  const StateIndex nStates;
  const OutputToken nOutToks;

  TraceDPMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const Trace& trace, const TraceParams& traceParams);

  void writeJson (ostream& out) const;
  string toJsonString() const;
  
  inline double& cell (OutputIndex outPos, StateIndex state) { return cellStorage[cellIndex(outPos,state)]; }
  inline const double cell (OutputIndex outPos, StateIndex state) const { return cellStorage[cellIndex(outPos,state)]; }

  inline double logEmitProb (OutputIndex outPos, OutputToken outTok) const {
    return coeffs.gauss[outTok-1].logEmitProb (moments.sample[outPos-1]);
  }
};

#endif /* DPTRACE_INCLUDED */

