#ifndef BACKTRACE_INCLUDED
#define BACKTRACE_INCLUDED

#include "forward.h"
#include "counts.h"

class BackwardTraceMatrix : public TraceDPMatrix {
private:
  vguard<IndexedTrans>::const_iterator nullTrans_rbegin, nullTrans_rend;
public:
  BackwardTraceMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const Trace& trace, const TraceParams& traceParams);
  
  void getMachineCounts (const TraceForwardMatrix&, MachineCounts&) const;
  void getGaussianCounts (const TraceForwardMatrix&, vguard<GaussianCounts>&) const;
  void getCounts (const TraceForwardMatrix&, GaussianModelCounts&) const;
  double logLike() const;
};

#endif /* BACKTRACE_INCLUDED */
