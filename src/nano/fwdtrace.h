#ifndef FWDTRACE_INCLUDED
#define FWDTRACE_INCLUDED

#include "dptrace.h"

struct ForwardTraceMatrix : TraceDPMatrix {
  ForwardTraceMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const Trace& trace, const TraceParams& traceParams);
  double logLike() const;
};

#endif /* FWDTRACE_INCLUDED */
