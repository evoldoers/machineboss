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

protected:
  vguard<vguard<IndexedTrans> > transByOut;  // indexed by output token
  inline const vguard<IndexedTrans>& nullTrans() const { return transByOut.front(); }
  size_t nTrans;

public:
  const EvaluatedMachine& eval;
  const GaussianModelParams& modelParams;
  const TraceMoments& moments;
  const TraceParams& traceParams;
  const GaussianModelCoefficients coeffs;
  const OutputIndex outLen;
  const StateIndex nStates;
  const OutputToken nOutToks;

  TraceDPMatrix (const EvaluatedMachine&, const GaussianModelParams&, const TraceMoments&, const TraceParams&);

  void writeJson (ostream& out) const;
  friend ostream& operator<< (ostream&, const TraceDPMatrix&);

  inline vguard<double>& column (OutputIndex outPos) { return columnStorage[outPos]; }
  inline const vguard<double>& column (OutputIndex outPos) const { return columnStorage[outPos]; }
  
  inline double& cell (OutputIndex outPos, StateIndex state) { return columnStorage[outPos][state]; }
  inline const double cell (OutputIndex outPos, StateIndex state) const { return columnStorage[outPos][state]; }

  inline double logEmitProb (OutputIndex outPos, OutputToken outTok) const {
    return coeffs.gauss[outTok-1].logEmitProb (moments.sample[outPos-1]);
  }
};

#endif /* DPTRACE_INCLUDED */

