#ifndef FWDTRACE_INCLUDED
#define FWDTRACE_INCLUDED

#include "dptrace.h"

struct ForwardTraceMatrix : TraceDPMatrix {
  ForwardTraceMatrix (const EvaluatedMachine&, const GaussianModelParams&, const TraceMoments&, const TraceParams&);
  double logLike() const;
};

#endif /* FWDTRACE_INCLUDED */
