#ifndef BACKTRACE_INCLUDED
#define BACKTRACE_INCLUDED

#include "forward.h"
#include "counts.h"

struct TraceBackwardMatrix : TraceDPMatrix {
  TraceBackwardMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const Trace& trace, const TraceParams& traceParams);
  void getMachineCounts (const TraceForwardMatrix&, MachineCounts&) const;
  void getGaussianCounts (const TraceForwardMatrix&, vguard<GaussianCounts>&) const;
  void getCounts (const TraceForwardMatrix&, GaussianModelCounts&) const;
  double logLike() const;
};

#endif /* BACKTRACE_INCLUDED */
