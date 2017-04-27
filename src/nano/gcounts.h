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
  map<string,double> prob;
  vguard<GaussianCounts> gauss;
  vguard<OutputSymbol> gaussSymbol;
  GaussianModelCounts();
  void init (const EvaluatedMachine&);
  double add (const Machine&, const EvaluatedMachine&, const GaussianModelParams&, const Trace&, const TraceParams&);  // returns log-likelihood
  void optimizeTraceParams (TraceParams&, const EvaluatedMachine&, const GaussianModelParams&, const GaussianModelPrior&) const;
  json asJson() const;
  void writeJson (ostream& out) const;
  static void optimizeModelParams (GaussianModelParams&, const TraceListParams&, const GaussianModelPrior&, const list<EvaluatedMachine>&, const list<GaussianModelCounts>&);
  static double expectedLogEmit (const GaussianModelParams&, const TraceListParams&, const GaussianModelPrior&, const list<GaussianModelCounts>&);
};

#endif /* GAUSSCOUNTS_INCLUDED */
