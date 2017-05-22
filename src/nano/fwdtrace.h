#ifndef FWDTRACE_INCLUDED
#define FWDTRACE_INCLUDED

#include <random>
#include "dptrace.h"

class ForwardTraceMatrix : public TraceDPMatrix {
private:
  void fillColumn (OutputIndex outPos);
  OutputIndex lastCheckpoint;

public:
  double logLike;

  ForwardTraceMatrix (const EvaluatedMachine&, const GaussianModelParams&, const TraceMoments&, const TraceParams&, size_t blockBytes = 0, double bandWidth = 1);
  void readyColumn (OutputIndex);
  MachinePath samplePath (const Machine&, mt19937&);
};

#endif /* FWDTRACE_INCLUDED */
