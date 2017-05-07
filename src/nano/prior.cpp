#include <gsl/gsl_randist.h>
#include "../logsumexp.h"
#include "prior.h"

WeightExpr Prior::logGammaExpr (const WeightExpr& rateParam, double count, double time) {
  return WeightAlgebra::subtract (WeightAlgebra::multiply (count, WeightAlgebra::logOf(rateParam)),
				  WeightAlgebra::multiply (time, rateParam));
}

double Prior::logNormalGammaProb (double mu, double tau, double mu0, double n_mu, double tau0, double n_tau) {
  const double alpha = n_tau / 2, beta = (n_tau - 1) / (2*tau0);
  return logGammaPdf (tau, alpha - 1, beta)
    + logGaussianPdf (mu, mu0, 1 / sqrt(n_mu*tau));
}

WeightExpr Prior::logNormalGammaExpr (const WeightExpr& muParam, const WeightExpr& tauParam, double mu0, double n_mu, double tau0, double n_tau) {
  const double alpha = n_tau / 2, beta = (n_tau - 1) / (2*tau0);
  auto add = WeightAlgebra::add, multiply = WeightAlgebra::multiply, subtract = WeightAlgebra::subtract, divide = WeightAlgebra::divide;
  auto logOf = WeightAlgebra::logOf;
  const WeightExpr muParam_minus_mu0 = subtract (muParam, mu0);
  // log(prior) = (alpha-0.5)*log(tau) - beta*tau - n_mu*tau*(mu-mu_0)^2/2
  //            = (n_tau-1)*log(tau)/2 - (n_tau-1)*tau/(2*tau_0) - n_mu*tau*(mu-mu_0)^2/2
  WeightExpr w = subtract (multiply (alpha - 0.5, logOf(tauParam)),
			   add (multiply (beta, tauParam),
				multiply (multiply (n_mu / 2, tauParam),
					  multiply (muParam_minus_mu0, muParam_minus_mu0))));
  return w;
}

double Prior::logNormalInvSquareGammaProb (double mu, double sigma, double mu0, double n_mu, double sigma0, double n_sigma) {
  return logNormalGammaProb (mu, 1/(sigma*sigma), mu0, n_mu, 1/(sigma0*sigma0), n_sigma);
}

WeightExpr Prior::logNormalInvSquareGammaExpr (const WeightExpr& muParam, const WeightExpr& sigmaParam, double mu0, double n_mu, double sigma0, double n_sigma) {
  auto add = WeightAlgebra::add, multiply = WeightAlgebra::multiply, subtract = WeightAlgebra::subtract, divide = WeightAlgebra::divide;
  auto logOf = WeightAlgebra::logOf;
  return logNormalGammaExpr (muParam, divide(1,multiply(sigmaParam,sigmaParam)), mu0, n_mu, 1/(sigma0*sigma0), n_sigma);
}

TraceParamsPrior::TraceParamsPrior()
  : scale(1), scaleCount(2),
    shift(0), shiftCount(1),
    rateCount(1), rateTime(1)
{ }

WeightExpr TraceParamsPrior::logTraceExpr (const WeightExpr& shiftParam, const WeightExpr& scaleParam) const {
  return logNormalInvSquareGammaExpr (shiftParam, scaleParam, shift, shiftCount, scale, scaleCount);
}

double TraceParamsPrior::logProb (const TraceListParams& traceListParams) const {
  double lp = 0;
  for (const auto& tp: traceListParams.params)
    lp += logNormalInvSquareGammaProb (tp.shift, tp.scale, shift, shiftCount, scale, scaleCount)
      + logGammaPdf (tp.rate, rateCount, rateTime);
  return lp;
}

double GaussianModelPrior::logProb (const GaussianModelParams& modelParams, const TraceListParams& traceListParams) const {
  return logProb(modelParams) + TraceParamsPrior::logProb (traceListParams);
}

double GaussianModelPrior::logProb (const GaussianModelParams& modelParams) const {
  Assert (cons.prob.size() == 0, "No free probability parameters allowed!");
  double lp = 0;
  for (const auto& g: gauss) {
    const auto& m = modelParams.gauss.at(g.first);
    const auto& p = g.second;
    lp += logNormalGammaProb (m.mu, m.tau, p.mu0, p.n_mu, p.tau0, p.n_tau);
  }
  for (const auto& n: cons.norm) {
    vguard<double> c, m;
    c.reserve (n.size());
    m.reserve (n.size());
    for (const auto& p: n) {
      c.push_back (count.defs.at(p));
      m.push_back (modelParams.prob.defs.at(p));
    }
    lp += logDirichletPdf (m, c);
  }
  for (const auto& r_g: gamma)
    lp += logGammaPdf (modelParams.rate.defs.at(r_g.first), r_g.second.count, r_g.second.time);
  return lp;
}
