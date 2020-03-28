#ifndef FITTER_INCLUDED
#define FITTER_INCLUDED

#include "machine.h"
#include "params.h"
#include "constraints.h"
#include "seqpair.h"

namespace MachineBoss {

struct MachineFitter {
  Machine machine;
  Constraints constraints;
  Params seed, constants;

  Constraints allConstraints() const;  // combines machine.cons and constraints
  Params fit (const SeqPairList&) const;
  Params fit (const SeqPairList&, size_t) const;
  Params fit (const SeqPairList&, const list<Envelope>&) const;
};

}  // end namespace

#endif /* FITTER_INCLUDED */

