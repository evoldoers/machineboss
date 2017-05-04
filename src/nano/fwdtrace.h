#ifndef FWDTRACE_INCLUDED
#define FWDTRACE_INCLUDED

#include "dptrace.h"

struct ForwardTraceMatrix : TraceDPMatrix {
  ForwardTraceMatrix (const EvaluatedMachine&, const GaussianModelParams&, const TraceMoments&, const TraceParams&, size_t blockBytes = 0);
  void fillColumn (OutputIndex outPos);
  void refillBlock (OutputIndex blockStart);
  double logLike;
};

#endif /* FWDTRACE_INCLUDED */
