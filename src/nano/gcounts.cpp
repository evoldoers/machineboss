#include "gcounts.h"
#include "minimize.h"
#include "backtrace.h"
#include "../logger.h"

GaussianCounts::GaussianCounts() : m0(0), m1(0), m2(0)
{ }

void GaussianCounts::inc (const SampleMoments& sampleMoments, const double postProb) {
  m0 += sampleMoments.m0 * postProb;
  m1 += sampleMoments.m1 * postProb;
  m2 += sampleMoments.m2 * postProb;
}

GaussianModelCounts::GaussianModelCounts()
{ }

void GaussianModelCounts::init (const EvaluatedMachine& m) {
  machine.init(m);
  gauss = vguard<GaussianCounts> (m.outputTokenizer.tok2sym.size() - 1);
}

double GaussianModelCounts::add (const EvaluatedMachine& m, const GaussianModelParams& mp, const Trace& t, const TraceParams& tp) {
  const ForwardTraceMatrix forward (m, mp, t, tp);
  const BackwardTraceMatrix backward (m, mp, t, tp);
  backward.getCounts (forward, *this);
  return forward.logLike();
}

void GaussianModelCounts::optimizeModelParams (GaussianModelParams& modelParams, const TraceListParams& traceListParams, const GaussianModelPrior& modelPrior, const list<Machine>& machine, const list<EvaluatedMachine>& eval, const list<GaussianModelCounts>& modelCountsList) {
  LogThisAt(5,"Optimizing model parameters" << endl);
  const auto gaussSymbol = extract_keys (modelParams.gauss);
  const size_t nSym = gaussSymbol.size();
  for (size_t n = 0; n < nSym; ++n) {
    const OutputSymbol& outSym = gaussSymbol[n];
    const GaussianPrior& prior = modelPrior.gauss.at(outSym);
    GaussianParams& params = modelParams.gauss.at(outSym);
    double coeff_log_tau = 0, coeff_tau_mu = 0, coeff_tau_mu2 = 0, coeff_tau = 0;
    auto countsIter = modelCountsList.begin();
    for (size_t m = 0; m < modelCountsList.size(); ++m) {
      const GaussianModelCounts& modelCounts = *(countsIter++);
      const TraceParams& trace = traceListParams.params[m];
      const GaussianCounts& counts = modelCounts.gauss[n];
      coeff_log_tau += counts.m0 / 2;
      coeff_tau_mu += counts.m1 / trace.scale - counts.m0 * trace.shift;
      coeff_tau_mu2 -= counts.m0 / 2;
      coeff_tau += counts.m1 * trace.shift / trace.scale - counts.m0 * trace.shift * trace.shift / 2 - counts.m2 * trace.scale * trace.scale / 2;
    }
    coeff_log_tau += (prior.n_tau - 1) / 2;
    coeff_tau_mu += prior.n_mu * prior.mu0;
    coeff_tau_mu2 -= prior.n_mu / 2;
    coeff_tau -= prior.n_mu * prior.mu0 * prior.mu0 / 2 + (prior.n_tau - 1) / (2 * prior.tau0);
    
    params.mu = -coeff_tau_mu / (2 * coeff_tau_mu2);
    params.tau = coeff_log_tau / (coeff_tau_mu * coeff_tau_mu / (4 * coeff_tau_mu2) - coeff_tau);

  }

  map<string,double> paramCount;
  auto machineIter = machine.begin();
  for (auto& modelCounts: modelCountsList) {
    const auto pc = modelCounts.machine.paramCounts (*(machineIter++), modelParams.prob);
    for (auto p_c: pc)
      paramCount[p_c.first] += p_c.second;
  }

  for (auto& norm: modelPrior.cons.norm) {
    double sum = 0;
    for (auto& p: norm)
      sum += (paramCount[p] += modelPrior.count.defs.at(p).get<double>());
    for (auto& p: norm)
      modelParams.prob.defs[p] = paramCount[p] / sum;
  }
}

void GaussianModelCounts::optimizeTraceParams (TraceParams& traceParams, const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const GaussianModelPrior& modelPrior) const {
  double coeff_log_scale = 0, coeff_1_over_scale = 0, coeff_scale2 = 0, coeff_shift = 0, coeff_shift2 = 0, coeff_shift_over_scale = 0;
  for (size_t n = 0; n < gauss.size(); ++n) {
    const GaussianCounts& counts = gauss[n];
    const OutputSymbol& outSym = eval.outputTokenizer.tok2sym[n+1];
    const GaussianPrior& prior = modelPrior.gauss.at(outSym);
    const GaussianParams& params = modelParams.gauss.at(outSym);
    coeff_log_scale -= counts.m0;
    coeff_1_over_scale += counts.m1 * params.tau * params.mu;
    coeff_scale2 -= counts.m2 * params.tau / 2;
    coeff_shift -= counts.m0 * params.tau * params.mu;
    coeff_shift2 -= counts.m0 * params.tau / 2;
    coeff_shift_over_scale += counts.m1 * params.tau;
  }

  auto add = WeightAlgebra::add, multiply = WeightAlgebra::multiply, subtract = WeightAlgebra::subtract, divide = WeightAlgebra::divide;
  auto logOf = WeightAlgebra::logOf;

  const string shiftParam("shift"), sqrtScaleParam("sqrtScale");
  const WeightExpr scaleParam = multiply (sqrtScaleParam, sqrtScaleParam);
  
  WeightExpr objective = add (add (add (multiply (coeff_log_scale, logOf (scaleParam)),
					divide (coeff_1_over_scale, scaleParam)),
				   add (multiply (coeff_scale2, multiply (scaleParam, scaleParam)),
					multiply (coeff_shift, shiftParam))),
			      add (add (multiply (coeff_shift2, multiply (shiftParam, shiftParam)),
					multiply (coeff_shift_over_scale, divide (shiftParam, scaleParam))),
				   modelPrior.logTraceExpr (shiftParam, scaleParam)));

  LogThisAt(5,"Optimizing scaling parameters" << endl);
  LogThisAt(7,"Objective function for trace scaling: " << WeightAlgebra::toString(objective,ParamDefs()) << endl);

  ParamDefs defs;
  defs[shiftParam] = traceParams.shift;
  defs[sqrtScaleParam] = sqrt (traceParams.scale);

  const Minimizer minimizer (subtract (0, objective));  // we want to maximize objective, i.e. minimize (-objective)
  const ParamDefs optDefs = minimizer.minimize (defs);

  traceParams.shift = defs[shiftParam].get<double>();
  traceParams.scale = defs[sqrtScaleParam].get<double>() * defs[sqrtScaleParam].get<double>();
}

double GaussianModelCounts::expectedLogEmit (const GaussianModelParams& modelParams, const TraceListParams& traceListParams, const GaussianModelPrior& modelPrior, const list<GaussianModelCounts>& modelCountsList) {
  const auto gaussSymbol = extract_keys (modelParams.gauss);
  double lp = modelPrior.logProb (modelParams, traceListParams);
  // expected log-likelihood = sum_kmers sum_reads m0*(-log(scale)+(1/2)log(tau)-(1/2)log(2*pi)-(tau/2)(mu+shift)^2) + m1*(tau/scale)*(mu+shift) - m2*(tau/2*(scale^2))
  const double log_sqrt_2pi = log(2*M_PI)/2;
  for (size_t n = 0; n < gaussSymbol.size(); ++n) {
    const OutputSymbol& outSym = gaussSymbol[n];
    const GaussianPrior& prior = modelPrior.gauss.at(outSym);
    const GaussianParams& params = modelParams.gauss.at(outSym);
    auto countsIter = modelCountsList.begin();
    for (size_t m = 0; m < traceListParams.params.size(); ++m) {
      const TraceParams& traceParams = traceListParams.params[m];
      const GaussianCounts& counts = (*(countsIter++)).gauss[n];
      const double m0 = counts.m0, m1 = counts.m1, m2 = counts.m2;
      const double tau = params.tau, mu = params.mu;
      const double shift = traceParams.shift, scale = traceParams.scale;
      lp += m0*(-log(scale) + 0.5*log(tau) - log_sqrt_2pi - (tau/2)*(mu+shift)*(mu+shift))
	+ m1*(tau/scale)*(mu+shift)
	- m2*(tau/2)*scale*scale;
    }
  }
  return lp;
}

json GaussianModelCounts::asJson() const {
  json j;
  j["machine"] = JsonWriter<MachineCounts>::toJson (machine);
  json jg = json::array();
  for (auto& g: gauss)
    jg.push_back (json::array ({ g.m0, g.m1, g.m2 }));
  j["gaussian"] = jg;
  return j;
}

void GaussianModelCounts::writeJson (ostream& out) const {
  out << asJson();
}
