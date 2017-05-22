#ifndef VITERBI_TRACE_INCLUDED
#define VITERBI_TRACE_INCLUDED

#include "dptrace.h"

class ViterbiTraceMatrix : public TraceDPMatrix {
private:
  inline void update (OutputIndex outPos, StateIndex state, double newLogLike, InputToken inTok) {
    double& ll = cell(outPos,state);
    ll = inTok ? max(ll,newLogLike) : log_sum_exp(ll,newLogLike);
  }

  void fillColumn (OutputIndex outPos);
  OutputIndex lastCheckpoint;

public:
  double logLike;

  ViterbiTraceMatrix (const EvaluatedMachine&, const GaussianModelParams&, const TraceMoments&, const TraceParams&, size_t blockBytes = 0, double bandWidth = 1);
  void readyColumn (OutputIndex);
  MachinePath path (const Machine&);
};

#endif /* VITERBI_TRACE_INCLUDED */
