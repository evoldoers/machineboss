#ifndef FORWARD_INCLUDED
#define FORWARD_INCLUDED

#include "dpmatrix.h"

struct ForwardMatrix : DPMatrix {
  ForwardMatrix (const EvaluatedMachine&, const vguard<InputToken>&, const vguard<OutputToken>&);
  double logLike() const;
};

#endif /* FORWARD_INCLUDED */
