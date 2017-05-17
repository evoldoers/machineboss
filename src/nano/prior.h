#ifndef PRIOR_INCLUDED
#define PRIOR_INCLUDED

#include "../weight.h"
#include "../constraints.h"
#include "trace.h"
#include "gaussian.h"

struct Prior {
  static WeightExpr logGammaExpr (const WeightExpr& rateParam, double count, double time);
  static double logGammaProb (double rate, double count, double time);
  
  static double logNormalGammaProb (double mu, double tau, double mu0, double n_mu, double tau0, double n_tau);
  static WeightExpr logNormalGammaExpr (const WeightExpr& muParam, const WeightExpr& tauParam, double mu0, double n_mu, double tau0, double n_tau);

  static double logNormalInvSquareGammaProb (double mu, double sigma, double mu0, double n_mu, double sigma0, double n_sigma);
  static WeightExpr logNormalInvSquareGammaExpr (const WeightExpr& muParam, const WeightExpr& sigmaParam, double mu0, double n_mu, double sigma0, double n_sigma);
};

struct TraceParamsPrior : Prior {
  double scale, scaleCount;
  double shift, shiftCount;
  double rateCount, rateTime;
  TraceParamsPrior();
  WeightExpr logTraceExpr (const WeightExpr& shiftParam, const WeightExpr& scaleParam) const;
  double logProb (const TraceListParams& traceListParams) const;
};

struct GaussianPrior {
  double mu0, n_mu, tau0, n_tau;
};

struct GammaPrior {
  double count, time;
};

struct GaussianModelPrior : TraceParamsPrior {
  map<string,GaussianPrior> gauss;  // hyperparameters for Normal-Gamma priors
  map<string,GammaPrior> gamma;  // hyperparameters for Gamma priors
  ParamAssign count;  // pseudocounts for Dirichlet & Beta priors
  Constraints cons;  // normalization constraints
  
  double logProb (const GaussianModelParams& modelParams) const;
  double logProb (const GaussianModelParams& modelParams, const TraceListParams& traceListParams) const;
};

#endif /* PRIOR_INCLUDED */
