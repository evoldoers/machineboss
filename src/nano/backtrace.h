#ifndef BACKTRACE_INCLUDED
#define BACKTRACE_INCLUDED

#include "fwdtrace.h"
#include "gcounts.h"

class BackwardTraceMatrix : public TraceDPMatrix {
private:
  vguard<IndexedTrans>::const_reverse_iterator nullTrans_rbegin, nullTrans_rend;
public:
  BackwardTraceMatrix (const EvaluatedMachine&, const GaussianModelParams&, const TraceMoments&, const TraceParams&);
  
  void getMachineCounts (const ForwardTraceMatrix&, MachineCounts&) const;
  void getGaussianCounts (const ForwardTraceMatrix&, vguard<GaussianCounts>&) const;
  double logLike() const;
};

#endif /* BACKTRACE_INCLUDED */
