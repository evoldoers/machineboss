#ifndef BACKTRACE_INCLUDED
#define BACKTRACE_INCLUDED

#include "fwdtrace.h"
#include "gcounts.h"

class BackwardTraceMatrix : public TraceDPMatrix {
private:
  vguard<IndexedTrans>::const_reverse_iterator nullTrans_rbegin, nullTrans_rend;
public:
  BackwardTraceMatrix (ForwardTraceMatrix&, MachineCounts* = NULL, vguard<GaussianCounts>* = NULL);
  double logLike() const;
};

#endif /* BACKTRACE_INCLUDED */
