#ifndef PRIOR_INCLUDED
#define PRIOR_INCLUDED

#include "../weight.h"
#include "../constraints.h"
#include "trace.h"
#include "gaussian.h"

// https://en.wikipedia.org/wiki/Normal-gamma_distribution#Interpretation_of_parameters
// If
//  (mu,tau) ~ NormalGamma(mu_mean,lambda,alpha,beta)
// then
//   mu ~ Normal(mean mu_mean,precision lambda*tau)
//  tau ~ Gamma(alpha,beta)
// which is the same as the posterior obtained for
//   mu_mean (= mu_mode = mu0) from n_mu=lambda samples
//  tau_mean = alpha/beta (tau_mode = (alpha-1)/beta = tau0) from n_tau=2*alpha samples
// Marginal standard deviations:
//   mu_sd = beta/(lambda*(alpha-1))
//  tau_sd = sqrt(alpha)/beta
// thus
//     n_tau = 2*(tau_mean / tau_sd)^2
//      n_mu = 1/(mu_sd * (tau_mean - tau_sd^2 / tau_mean))
//   mu_mode = mu_mean
//  tau_mode = tau_mean * (1 - 2 / n_tau)
// also
//     alpha = n_tau / 2
//      beta = (n_tau - 1) / tau_mode
// For n_mu and tau_mode to be positive, we require tau_mean > tau_sd
// If tau_mean = sigma_mean^{-2} and tau_sd = sigma_sd^{-2},
// then this condition implies sigma_mean < sigma_sd

#define CALC_N_TAU(TAU_MEAN,TAU_SD) (2. * ((TAU_MEAN) / (double) (TAU_SD)) * ((TAU_MEAN) / (double) (TAU_SD)))
#define CALC_N_MU(MU_SD,TAU_MEAN,TAU_SD) (1. / ((MU_SD) * ((TAU_MEAN) - (TAU_SD) * (TAU_SD) / (double) (TAU_MEAN))))
#define CALC_TAU_MODE(TAU_MEAN,TAU_SD) (TAU_MEAN * (1. - 2. / CALC_N_TAU(TAU_MEAN,TAU_SD)))
#define CALC_TAU_MEAN(SIGMA_MEAN) (1. / SIGMA_MEAN / SIGMA_MEAN)
#define CALC_TAU_SD(SIGMA_SD) (1. / SIGMA_SD / SIGMA_SD)

struct Prior {
  static WeightExpr logGammaExpr (const WeightExpr& rateParam, double count, double time);
  static double logGammaProb (double rate, double count, double time);

  // NB: tau0 = tau_mode, mu0 = mu_mode
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
