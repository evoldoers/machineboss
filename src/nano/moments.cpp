#include <math.h>
#include "../util.h"
#include "moments.h"

TraceMoments::TraceMoments (const Trace& trace)
  : sample (trace.sample.size())
{
  for (TraceIndex j = 0; j < sample.size(); ++j) {
    const double t = trace.sample[j];
    SampleMoments& e = sample[j];
    e.m0 = 1;
    e.m1 = t;
    e.m2 = t*t;
  }
}

GaussianModelCoefficients::GaussianModelCoefficients (const GaussianModelParams& modelParams, const TraceParams& traceParams, const OutputTokenizer& outputTokenizer)
  : gauss (outputTokenizer.tok2sym.size() - 1)
{
  const double log_sqrt_2pi = log(2*M_PI)/2;
  const double shift = traceParams.shift, scale = traceParams.scale;
  for (OutputToken outTok = 1; outTok <= outputTokenizer.tok2sym.size(); ++outTok) {
    const auto& outSym = outputTokenizer.tok2sym[outTok];
    const GaussianParams& p = modelParams.gaussian.at(outSym);
    GaussianCoefficients& e = gauss[outTok-1];
    const double mu_plus_shift = p.mu + shift;
    e.m0coeff = log(p.tau)/2 - log(scale) - log_sqrt_2pi - (p.tau/2)*mu_plus_shift*mu_plus_shift;
    e.m1coeff = (p.tau/scale) * mu_plus_shift;
    e.m2coeff = -p.tau/(2*scale*scale);
  }
}
    
