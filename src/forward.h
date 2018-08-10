#ifndef FORWARD_INCLUDED
#define FORWARD_INCLUDED

#include "dpmatrix.h"

class ForwardMatrix : public DPMatrix {
private:
  void fill();
public:
  ForwardMatrix (const EvaluatedMachine&, const SeqPair&);
  ForwardMatrix (const EvaluatedMachine&, const SeqPair&, const Envelope&);
  double logLike() const;
};

#endif /* FORWARD_INCLUDED */
