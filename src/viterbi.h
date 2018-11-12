#ifndef VITERBI_INCLUDED
#define VITERBI_INCLUDED

#include "dpmatrix.h"

class ViterbiMatrix : public DPMatrix {
private:
  void fill();
  
public:
  ViterbiMatrix (const EvaluatedMachine&, const SeqPair&);
  ViterbiMatrix (const EvaluatedMachine&, const SeqPair&, const Envelope&);
  double logLike() const;
  MachinePath path (const Machine&) const;
};

#endif /* VITERBI_INCLUDED */
