#ifndef FITTER_INCLUDED
#define FITTER_INCLUDED

#include "counts.h"
#include "jsonio.h"

struct MachineFitterBase {
  Params seed;
  Constraints constraints;
  SeqPairList trainingSet;

  void readJson (const json&);
  void writeJson (ostream&);

  Params fit (const Machine&) const;
};

typedef JsonLoader<MachineFitterBase> MachineFitter;

#endif /* FITTER_INCLUDED */

