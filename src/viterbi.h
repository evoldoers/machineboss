#ifndef VITERBI_INCLUDED
#define VITERBI_INCLUDED

#include "dpmatrix.h"

namespace MachineBoss {

class ViterbiMatrix : public DPMatrix<IdentityIndexMapper> {
private:
  void fill();
  
public:
  ViterbiMatrix (const EvaluatedMachine&, const SeqPair&);
  ViterbiMatrix (const EvaluatedMachine&, const SeqPair&, const Envelope&);
  double logLike() const;
  MachinePath path (const Machine&) const;
};

}  // end namespace

#endif /* VITERBI_INCLUDED */
