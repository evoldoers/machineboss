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
  prob.clear();
  gaussIndex.clear();
  gauss = vguard<GaussianCounts> (m.outputTokenizer.tok2sym.size() - 1);
  for (size_t n = 1; n < m.outputTokenizer.tok2sym.size(); ++n)
    gaussIndex[m.outputTokenizer.tok2sym[n]] = n - 1;
}

double GaussianModelCounts::add (const Machine& machine, const EvaluatedMachine& m, const GaussianModelParams& mp, const TraceMoments& t, const TraceParams& tp, size_t blockBytes, double bandWidth) {
  MachineCounts mc (m);
  ForwardTraceMatrix forward (m, mp, t, tp, blockBytes, bandWidth);
  const BackwardTraceMatrix backward (forward, &mc, &gauss);

  const auto pc = mc.paramCounts (machine, mp.params (tp.rate));
  for (auto p_c: pc)
    prob[p_c.first] += p_c.second;

  return forward.logLike;
}

void GaussianModelCounts::optimizeModelParams (GaussianModelParams& modelParams, const TraceListParams& traceListParams, const GaussianModelPrior& modelPrior, const list<EvaluatedMachine>& eval, const list<GaussianModelCounts>& modelCountsList) {
  LogThisAt(5,"Optimizing model parameters" << endl);
  const auto gaussSymbol = extract_keys (modelParams.gauss);
  const size_t nSym = gaussSymbol.size();
  for (size_t n = 0; n < nSym; ++n) {
    const OutputSymbol& outSym = gaussSymbol[n];
    const GaussianPrior& prior = modelPrior.gauss.at(outSym);
    GaussianParams& params = modelParams.gauss.at(outSym);
    // expected log-likelihood = sum_datasets m0*(-log(scale)+(1/2)log(tau)-(1/2)log(2*pi)-(tau/2)(mu+shift)^2) + m1*(tau/scale)*(mu+shift) - m2*tau/(2*(scale^2))
    double coeff_log_tau = 0, coeff_tau_mu = 0, coeff_tau_mu2 = 0, coeff_tau = 0;
    auto countsIter = modelCountsList.begin();
    for (size_t m = 0; m < modelCountsList.size(); ++m) {
      const GaussianModelCounts& modelCounts = *(countsIter++);
      if (modelCounts.gaussIndex.count(outSym)) {
	const GaussianCounts& counts = modelCounts.gauss[modelCounts.gaussIndex.at(outSym)];
	const TraceParams& trace = traceListParams.params[m];
	coeff_log_tau += counts.m0 / 2;
	coeff_tau_mu += counts.m1 / trace.scale - counts.m0 * trace.shift;
	coeff_tau_mu2 -= counts.m0 / 2;
	coeff_tau += counts.m1 * trace.shift / trace.scale - counts.m0 * trace.shift * trace.shift / 2 - counts.m2 / (2 * trace.scale * trace.scale);
      }
    }

    coeff_log_tau += (prior.n_tau - 1) / 2;
    coeff_tau_mu += prior.n_mu * prior.mu0;
    coeff_tau_mu2 -= prior.n_mu / 2;
    coeff_tau -= prior.n_mu * prior.mu0 * prior.mu0 / 2 + (prior.n_tau - 1) / (2 * prior.tau0);
    
    params.mu = -coeff_tau_mu / (2 * coeff_tau_mu2);
    params.tau = coeff_log_tau / (coeff_tau_mu * coeff_tau_mu / (4 * coeff_tau_mu2) - coeff_tau);
  }

  map<string,double> paramCount;
  for (auto& modelCounts: modelCountsList)
    for (auto p_c: modelCounts.prob)
      paramCount[p_c.first] += p_c.second;

  for (auto& norm: modelPrior.cons.norm) {
    double sum = 0;
    for (auto& p: norm)
      sum += (paramCount[p] += modelPrior.count.defs.at(p).get<double>());
    for (auto& p: norm)
      modelParams.prob.defs[p] = paramCount[p] / sum;
  }

  for (auto& rateParam_gamma: modelPrior.gamma) {
    const string& rateParam = rateParam_gamma.first;
    const GammaPrior& gamma = rateParam_gamma.second;
    auto countsIter = modelCountsList.begin();
    double time = 0, count = 0;
    for (size_t m = 0; m < modelCountsList.size(); ++m) {
      const GaussianModelCounts& modelCounts = *(countsIter++);
      const TraceParams& trace = traceListParams.params[m];
      time += modelCounts.eventWait (rateParam, modelParams, trace) * trace.rate;
      count += modelCounts.eventCount (rateParam);
    }
    modelParams.rate.defs[rateParam] = (count + gamma.count) / (time + gamma.time);
  }
}

double GaussianModelCounts::eventCount (const string& rateParam) const {
  const string exitEvent = GaussianModelParams::exitEventFuncName (rateParam);
  return prob.count(exitEvent) ? prob.at(exitEvent) : 0.;
}

double GaussianModelCounts::eventWait (const string& rateParam, const GaussianModelParams& modelParams, const TraceParams& traceParams) const {
  const string exitEvent = GaussianModelParams::exitEventFuncName (rateParam);
  const string waitEvent = GaussianModelParams::waitEventFuncName (rateParam);
  double t = 0;
  if (prob.count(waitEvent))
    t += prob.at(waitEvent);
  if (prob.count(exitEvent)) {
    const double r = modelParams.rate.defs.at(rateParam).get<double>() * traceParams.rate;
    const double e = exp(-r);
    t += prob.at(exitEvent) * (1-(r+1)*e) / (r*(1-e));  // expected amount of time before exit event occurred
  }
  return t;
}

WeightExpr GaussianModelCounts::traceExpectedLogEmit (const GaussianModelParams& modelParams, const GaussianModelPrior& modelPrior) const {
  // expected log-likelihood = sum_gaussians m0*(-log(scale)+(1/2)log(tau)-(1/2)log(2*pi)-(tau/2)(mu+shift)^2) + m1*(tau/scale)*(mu+shift) - m2*tau/(2*(scale^2))
  double coeff_log_scale = 0, coeff_1_over_scale = 0, coeff_1_over_scale2 = 0, coeff_shift = 0, coeff_shift2 = 0, coeff_shift_over_scale = 0;
  for (auto& outSym_n: gaussIndex) {
    const OutputSymbol& outSym = outSym_n.first;
    const size_t n = outSym_n.second;
    const GaussianCounts& counts = gauss[n];
    const GaussianPrior& prior = modelPrior.gauss.at(outSym);
    const GaussianParams& params = modelParams.gauss.at(outSym);
    coeff_log_scale -= counts.m0;
    coeff_1_over_scale += counts.m1 * params.tau * params.mu;
    coeff_1_over_scale2 -= counts.m2 * params.tau / 2;
    coeff_shift -= counts.m0 * params.tau * params.mu;
    coeff_shift2 -= counts.m0 * params.tau / 2;
    coeff_shift_over_scale += counts.m1 * params.tau;
  }

  auto add = WeightAlgebra::add, multiply = WeightAlgebra::multiply, divide = WeightAlgebra::divide;
  auto logOf = WeightAlgebra::logOf;

  const WeightExpr shiftParam = shiftParamName();
  const WeightExpr scaleParam = multiply (sqrtScaleParamName(), sqrtScaleParamName());
  
  const WeightExpr e = add (add (add (multiply (coeff_log_scale, logOf (scaleParam)),
				      divide (coeff_1_over_scale, scaleParam)),
				 add (divide (coeff_1_over_scale2, multiply (scaleParam, scaleParam)),
				      multiply (coeff_shift, shiftParam))),
			    add (add (multiply (coeff_shift2, multiply (shiftParam, shiftParam)),
				      multiply (coeff_shift_over_scale, divide (shiftParam, scaleParam))),
				 modelPrior.logTraceExpr (shiftParam, scaleParam)));

  return e;
}

ParamDefs GaussianModelCounts::traceEmitParamDefs (const TraceParams& traceParams) {
  ParamDefs defs;
  defs[shiftParamName()] = traceParams.shift;
  defs[sqrtScaleParamName()] = sqrt (traceParams.scale);
  return defs;
}

double GaussianModelCounts::traceExpectedLogEvents (const GaussianModelParams& modelParams, const TraceParams& traceParams, const GaussianModelPrior& modelPrior) const {
  double lp = modelPrior.logGammaProb (traceParams.rate, modelPrior.rateCount, modelPrior.rateTime);
  for (const auto& param_rate: modelParams.rate.defs) {
    const string& rateParam = param_rate.first;
    const double waitProb = exp (-traceParams.rate * param_rate.second.get<double>());
    const string waitEvent = GaussianModelParams::waitEventFuncName (rateParam);
    const string exitEvent = GaussianModelParams::exitEventFuncName (rateParam);
    if (prob.count(waitEvent))
      lp += prob.at(waitEvent) * log(waitProb);
    if (prob.count(exitEvent))
      lp += prob.at(exitEvent) * log(1. - waitProb);
    }
  return lp;
}

void GaussianModelCounts::optimizeTraceParams (TraceParams& traceParams, const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const GaussianModelPrior& modelPrior) const {
  LogThisAt(5,"Optimizing trace scaling parameters" << endl);

  const WeightExpr emitObjective = traceExpectedLogEmit (modelParams, modelPrior);
  const Minimizer emitMinimizer (WeightAlgebra::subtract (false, emitObjective));  // we want to maximize objective, i.e. minimize (-objective)
  const ParamDefs seedEmitDefs = traceEmitParamDefs (traceParams);
  const ParamDefs optEmitDefs = emitMinimizer.minimize (seedEmitDefs);

  traceParams.shift = optEmitDefs.at(shiftParamName()).get<double>();
  traceParams.scale = pow (optEmitDefs.at(sqrtScaleParamName()).get<double>(), 2.);

  double count = 0, time = 0;
  for (const auto& param_rate: modelParams.rate.defs) {
    const string& rateParam = param_rate.first;
    count += eventCount (rateParam);
    time += eventWait (rateParam, modelParams, traceParams) * param_rate.second.get<double>();
  }
  traceParams.rate = (count + modelPrior.rateCount) / (time + modelPrior.rateTime);
}

double GaussianModelCounts::expectedLogLike (const GaussianModelParams& modelParams, const TraceListParams& traceListParams, const GaussianModelPrior& modelPrior, const list<GaussianModelCounts>& modelCountsList) {
  const auto gaussSymbol = extract_keys (modelParams.gauss);
  double lp = modelPrior.logProb (modelParams);
  size_t m = 0;
  for (const GaussianModelCounts& modelCounts: modelCountsList) {
    const TraceParams& traceParams = traceListParams.params[m++];
    const ParamDefs emitDefs = traceEmitParamDefs (traceParams);
    const WeightExpr logEmit = modelCounts.traceExpectedLogEmit (modelParams, modelPrior);
    lp += WeightAlgebra::eval (logEmit, emitDefs);
    lp += modelCounts.traceExpectedLogEvents (modelParams, traceParams, modelPrior);
  }
  return lp;
}

json GaussianModelCounts::asJson() const {
  json jp = json::object();
  for (auto& p_v: prob)
    jp[p_v.first] = p_v.second;

  const auto gaussSymbol = extract_keys(gaussIndex);
  json jg = json::object();
  for (size_t n = 0; n < gauss.size(); ++n)
    jg[gaussSymbol[n]] = json::array ({ gauss[n].m0, gauss[n].m1, gauss[n].m2 });

  return json::object ({ { "gaussian", jg }, { "prob", jp } });
}

void GaussianModelCounts::writeJson (ostream& out) const {
  out << asJson() << endl;
}
