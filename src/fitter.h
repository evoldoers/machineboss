#ifndef FITTER_INCLUDED
#define FITTER_INCLUDED

#include "machine.h"
#include "params.h"
#include "constraints.h"
#include "seqpair.h"
#include "jsonio.h"

struct MachineFitterBase {
  Machine machine;
  Constraints constraints;
  Params seed;
  
  void readJson (const json&);
  void writeJson (ostream&) const;

  Params fit (const SeqPairList& trainingSet) const;
};

typedef JsonLoader<MachineFitterBase> MachineFitter;

#endif /* FITTER_INCLUDED */

