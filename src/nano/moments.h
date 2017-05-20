#ifndef MOMENTS_INCLUDED
#define MOMENTS_INCLUDED

#include "../eval.h"
#include "gaussian.h"
#include "trace.h"

struct SampleMoments {
  int m0;
  double m1, m2;
  SampleMoments();
  SampleMoments (const Trace&, size_t pos, size_t len);
};

struct TraceMoments {
  string name;
  vguard<SampleMoments> sample;
  TraceMoments();
  TraceMoments (const Trace& trace);

  string pathScoreBreakdown (const Machine&, const MachinePath&, const GaussianModelParams&, const TraceParams&) const;
};

struct TraceMomentsList {
  list<TraceMoments> trace;
  inline size_t size() const { return trace.size(); }
  TraceMomentsList();
  TraceMomentsList (const TraceList&);
  TraceMomentsList (const TraceList&, double maxFracDiff, size_t maxSegLen);
};

struct GaussianCoefficients {
  double m0coeff, m1coeff, m2coeff;  // coefficients of m0, m1, m2 in log-likelihood
  inline const double logEmitProb (const SampleMoments& x) const {
    return x.m0 * m0coeff + x.m1 * m1coeff + x.m2 * m2coeff;
  }
};

struct GaussianModelCoefficients {
  vguard<GaussianCoefficients> gauss;
  GaussianModelCoefficients (const GaussianModelParams& modelParams, const TraceParams& traceParams, const OutputTokenizer& outputTokenizer);
};

#endif /* MOMENTS_INCLUDED */
