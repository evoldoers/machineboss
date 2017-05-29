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

  size_t blockBytes;
  double bandWidth;
  bool fitTrace;
  
  GaussianTrainer();
  void init (const Machine&, const GaussianModelParams&, const GaussianModelPrior&, const TraceMomentsList&);
  void reset();
  bool testFinished();
  double expectedLogLike() const;
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

struct DummyGenerator : Machine {
  DummyGenerator (const vguard<InputSymbol>& alphabet);
};

#endif /* GAUSSTRAINER_INCLUDED */
