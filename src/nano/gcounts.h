#ifndef GAUSSCOUNTS_INCLUDED
#define GAUSSCOUNTS_INCLUDED

#include "../counts.h"
#include "moments.h"

struct GaussianCounts {
  double m0, m1, m2;
  GaussianCounts();
};

struct GaussianModelCounts {
  MachineCounts machine;
  vguard<GaussianCounts> gauss;
  void init (const EvaluatedMachine&);
  double add (const EvaluatedMachine&, const GaussianModelParams&, const Trace&, const TraceParams&);  // returns log-likelihood
  void optimizeModelParams (GaussianModelParams&, const TraceListParams&, const GaussianModelPrior&) const;
  void optimizeTraceParams (TraceParams&, const GaussianModelParams&, const GaussianModelPrior&) const;
  double expectedLogEmit (const GaussianModelParams&, const TraceListParams&, const GaussianModelPrior&) const;
};

#endif /* GAUSSCOUNTS_INCLUDED */
