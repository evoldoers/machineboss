#ifndef FITTER_INCLUDED
#define FITTER_INCLUDED

#include "machine.h"
#include "params.h"
#include "constraints.h"
#include "seqpair.h"

struct MachineFitter {
  Machine machine;
  Constraints constraints;
  Params seed, constants;

  Params fit (const SeqPairList& trainingSet) const;
};

#endif /* FITTER_INCLUDED */

