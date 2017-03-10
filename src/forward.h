#ifndef FORWARD_INCLUDED
#define FORWARD_INCLUDED

#include "dpmatrix.h"

struct ForwardMatrix : DPMatrix {
  ForwardMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair);
  double logLike() const;
};

#endif /* FORWARD_INCLUDED */
