#ifndef FWDTRACE_INCLUDED
#define FWDTRACE_INCLUDED

#include <random>
#include "dptrace.h"

class ForwardTraceMatrix : public TraceDPMatrix {
private:
  void fillColumn (OutputIndex outPos);
  OutputIndex lastCheckpoint;

public:
  ForwardTraceMatrix (const EvaluatedMachine&, const GaussianModelParams&, const TraceMoments&, const TraceParams&, size_t blockBytes = 0);
  void readyColumn (OutputIndex);
  MachinePath samplePath (const Machine&, mt19937&);
  double logLike;
};

#endif /* FWDTRACE_INCLUDED */
