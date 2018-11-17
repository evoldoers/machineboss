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
  MachinePath samplePath (const Machine&, mt19937&) const;
};

#endif /* FORWARD_INCLUDED */
