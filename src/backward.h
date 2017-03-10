#ifndef BACKWARD_INCLUDED
#define BACKWARD_INCLUDED

#include "forward.h"
#include "counts.h"

struct BackwardMatrix : DPMatrix {
  BackwardMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair);
  void getCounts (const ForwardMatrix&, MachineCounts&) const;
  double logLike() const;
};

#endif /* BACKWARD_INCLUDED */
