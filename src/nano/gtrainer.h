#ifndef GAUSSTRAINER_INCLUDED
#define GAUSSTRAINER_INCLUDED

#include "prior.h"

struct GaussianTrainer {
  Machine machine;
  GaussianModelPrior prior;

  TraceList traceList;

  TraceListParams traceListParams;
  GaussianModelParams modelParams;

  GaussianModelCounts counts;
  double logPrior, logLike, prevLogLike;  // logLike includes logPrior
  size_t iter;
  
  void init (const Machine&, const GaussianModelParams&, const TraceList&);
  void reset();
  bool testFinished();
};

struct GaussianModelFitter : GaussianTrainer {
  vguard<FastSeq> seqs;
  vguard<Machine> inputConditionedMachine;
  
  void init (const Machine&, const GaussianModelParams&, const TraceList&, const vguard<FastSeq>&);
  void fit();
};

struct GaussianDecoder : GaussianTrainer {
  vguard<FastSeq> decode();
};

#endif /* GAUSSTRAINER_INCLUDED */
