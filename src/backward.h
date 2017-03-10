#ifndef BACKWARD_INCLUDED
#define BACKWARD_INCLUDED

#include "dpmatrix.h"

struct BackwardMatrix : DPMatrix {
  BackwardMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair);
  double logLike() const;
};

#endif /* BACKWARD_INCLUDED */
