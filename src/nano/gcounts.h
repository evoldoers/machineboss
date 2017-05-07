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

// y = scale * (x + shift)
// x ~ Normal(mean mu, precision tau)
// P(y) = P(x) dx/dy = P(x) / scale
// log P(y) = -log(scale) - log(Normal(y/scale-shift,mu,tau))
//          = -log(scale) + (1/2)*log(tau/(2*pi)) - (tau/2)*(y/scale-shift-mu)^2
//          = -log(scale) + (1/2)*log(tau/(2*pi)) - (tau/2)*((y/scale)^2 - 2*(y/scale)*(mu+shift) + (mu+shift)^2)

// Expected log-likelihood
//  = sum_gaussians sum_datasets sum_samples log P(y)
//  = sum_gaussians sum_datasets m0*(-log(scale)+(1/2)log(tau)-(1/2)log(2*pi)-(tau/2)(mu+shift)^2) + m1*(tau/scale)*(mu+shift) - m2*tau/(2*(scale^2))

struct GaussianModelCounts {
  map<string,double> prob;
  vguard<GaussianCounts> gauss;
  map<OutputSymbol,size_t> gaussIndex;
  GaussianModelCounts();
  void init (const EvaluatedMachine&);
  double add (const Machine&, const EvaluatedMachine&, const GaussianModelParams&, const TraceMoments&, const TraceParams&, size_t blockBytes = 0);  // returns log-likelihood
  WeightExpr traceExpectedLogEmit (const GaussianModelParams&, const GaussianModelPrior&) const;
  WeightExpr traceExpectedLogEvents (const EventMachine&, const GaussianModelParams&, const GaussianModelPrior&) const;
  void optimizeTraceParams (TraceParams&, const EventMachine&, const EvaluatedMachine&, const GaussianModelParams&, const GaussianModelPrior&) const;
  json asJson() const;
  void writeJson (ostream& out) const;
  static void optimizeModelParams (GaussianModelParams&, const TraceListParams&, const GaussianModelPrior&, const EventMachine&, const list<EvaluatedMachine>&, const list<GaussianModelCounts>&);
  static double expectedLogLike (const EventMachine&, const GaussianModelParams&, const TraceListParams&, const GaussianModelPrior&, const list<GaussianModelCounts>&);
  static ParamDefs traceEventParamDefs (const TraceParams&);
  static ParamDefs traceEmitParamDefs (const TraceParams&);
  static inline string shiftParamName() { return string("shift"); }
  static inline string sqrtScaleParamName() { return string("sqrtScale"); }
  static inline string sqrtRateParamName() { return string("sqrtRate"); }
};

#endif /* GAUSSCOUNTS_INCLUDED */
