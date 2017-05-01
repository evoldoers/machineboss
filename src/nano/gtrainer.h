#ifndef GAUSSTRAINER_INCLUDED
#define GAUSSTRAINER_INCLUDED

#include "../fastseq.h"
#include "gcounts.h"

struct GaussianTrainer {
  Machine machine;
  GaussianModelPrior prior;

  TraceMomentsList traceList;

  TraceListParams traceListParams;
  GaussianModelParams modelParams;

  list<GaussianModelCounts> counts;
  double logPrior, logLike, prevLogLike;  // logLike includes logPrior
  size_t iter;
  
  void init (const Machine&, const GaussianModelParams&, const GaussianModelPrior&, const TraceMomentsList&);
  void reset();
  bool testFinished();
  double expectedLogEmit() const;
};

struct GaussianModelFitter : GaussianTrainer {
  vguard<FastSeq> seqs;
  list<Machine> inputConditionedMachine;
  
  void init (const Machine&, const GaussianModelParams&, const GaussianModelPrior&, const TraceMomentsList&, const vguard<FastSeq>&);
  void fit();
};

struct GaussianDecoder : GaussianTrainer {
  vguard<FastSeq> decode();
};

#endif /* GAUSSTRAINER_INCLUDED */
