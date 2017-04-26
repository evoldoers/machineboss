#ifndef VITERBI_TRACE_INCLUDED
#define VITERBI_TRACE_INCLUDED

#include "dptrace.h"

class ViterbiTraceMatrix : public TraceDPMatrix {
private:
  inline void pathIterate (double& bestLogLike, StateIndex& bestSource, EvaluatedMachineState::TransIndex& bestTransIndex, const EvaluatedMachineState::InOutStateTransMap& incoming, OutputIndex outPos) const {
  }

  inline void update (OutputIndex outPos, StateIndex state, double newLogLike, InputToken inTok) {
    double& ll = cell(outPos,state);
    ll = inTok ? max(ll,newLogLike) : log_sum_exp(ll,newLogLike);
  }

public:
  ViterbiTraceMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const Trace& trace, const TraceParams& traceParams);
  double logLike() const;
  MachinePath path (const Machine&) const;
};

#endif /* VITERBI_TRACE_INCLUDED */
