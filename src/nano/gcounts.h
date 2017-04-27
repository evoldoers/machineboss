#ifndef GAUSSCOUNTS_INCLUDED
#define GAUSSCOUNTS_INCLUDED

#include "../counts.h"
#include "moments.h"
#include "prior.h"

struct GaussianCounts {
  double m0, m1, m2;
  GaussianCounts();
  void inc (const SampleMoments& sampleMoments, const double postProb);
};

struct GaussianModelCounts {
  MachineCounts machine;
  vguard<GaussianCounts> gauss;
  void init (const EvaluatedMachine&);
  double add (const EvaluatedMachine&, const GaussianModelParams&, const Trace&, const TraceParams&);  // returns log-likelihood
  void optimizeTraceParams (TraceParams&, const EvaluatedMachine&, const GaussianModelParams&, const GaussianModelPrior&) const;
  GaussianModelCounts& operator+= (const GaussianModelCounts&);
  static void optimizeModelParams (GaussianModelParams&, const TraceListParams&, const GaussianModelPrior&, const list<Machine>&, const list<EvaluatedMachine>&, const list<GaussianModelCounts>&);
  static double expectedLogEmit (const GaussianModelParams&, const TraceListParams&, const GaussianModelPrior&, const list<GaussianModelCounts>&);
};

#endif /* GAUSSCOUNTS_INCLUDED */
