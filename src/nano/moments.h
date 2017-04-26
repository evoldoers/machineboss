#ifndef MOMENTS_INCLUDED
#define MOMENTS_INCLUDED

#include "../eval.h"
#include "gaussian.h"
#include "trace.h"

struct SampleMoments {
  double m0, m1, m2;
};

struct TraceMoments {
  vguard<SampleMoments> sample;
  TraceMoments (const Trace& trace);
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
