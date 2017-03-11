#ifndef COUNTS_INCLUDED
#define COUNTS_INCLUDED

#include "eval.h"
#include "seqpair.h"
#include "constraints.h"

// E-step
struct MachineCounts {
  vguard<vguard<double> > count;  // indexed: count[state][nTrans]
  MachineCounts (const EvaluatedMachine&, const SeqPair&);
  MachineCounts& operator+= (const MachineCounts&);
  void writeJson (ostream&) const;
};

// M-step
struct MachineLagrangian {
  vguard<string> param;
  map<string,size_t> transformedParamIndex;
  ParamDefs paramTransform;
  WeightExpr lagrangian, gradSquared;
  vguard<WeightExpr> deriv;
  MachineLagrangian (const Machine&, const MachineCounts&, const Constraints&);
  Params optimize (const Params& seed) const;
};

#endif /* COUNTS_INCLUDED */

