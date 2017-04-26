#ifndef DPTRACE_INCLUDED
#define DPTRACE_INCLUDED

#include "../eval.h"
#include "../logsumexp.h"
#include "trace.h"

class TraceDPMatrix {
public:
  typedef long OutputIndex;

private:
  typedef long long CellIndex;
  typedef long long EmitIndex;

  vguard<double> cellStorage;

  inline CellIndex nCells() const {
    return nStates * (outLen + 1);
  }

  inline CellIndex cellIndex (OutputIndex outPos, StateIndex state) const {
    return outPos * nGaussians + state;
  }

public:
  const EvaluatedMachine& eval;
  const GaussianModelParams& modelParams;
  const Trace& trace;
  const TraceParams& traceParams;
  const TraceMoments moments;
  const GaussianModelCoefficients coeffs;
  const OutputIndex outLen;
  const StateIndex nStates;
  const GaussianIndex nGaussians;
  
public:
  TraceDPMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const Trace& trace, const TraceParams& traceParams);

  void writeJson (ostream& out) const;
  string toJsonString() const;
  
  inline double& cell (OutputIndex outPos, StateIndex state) { return cellStorage[cellIndex(outPos,state)]; }
  inline const double cell (OutputIndex outPos, StateIndex state) const { return cellStorage[cellIndex(outPos,state)]; }

  inline double logEmitProb (OutputIndex outPos, GaussianIndex gaussian) {
    return coeffs.gauss[gaussian].logEmitProb (moments.sample[outPos-1]);
  }
};

#endif /* DPTRACE_INCLUDED */

