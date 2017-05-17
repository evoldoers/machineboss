#include "gaussian.h"

string EventFuncNamer::waitEventFuncName (const string& rateParam) {
  return string("exp(-") + rateParam + "*t)";
}

string EventFuncNamer::exitEventFuncName (const string& rateParam) {
  return string("1-exp(-") + rateParam + "*t)";
}

GaussianParams::GaussianParams() :
  mu (0),
  tau (1)
{ }

json GaussianModelParams::asJson() const {
  json jg = json::object();
  for (auto& g: gauss)
    jg[g.first] = json ({ {"mu", g.second.mu}, {"sigma", 1./sqrt(g.second.tau)} });
  return json::object ({ { "gauss", jg },
	{ "prob", JsonWriter<ParamAssign>::toJson(prob) },
	  { "rate", JsonWriter<ParamAssign>::toJson(rate) } });
}

void GaussianModelParams::writeJson (ostream& out) const {
  out << asJson() << endl;
}

void GaussianModelParams::readJson (const json& j) {
  gauss.clear();
  prob.clear();
  const json& jg = j["gauss"];
  for (json::const_iterator gaussIter = jg.begin(); gaussIter != jg.end(); ++gaussIter) {
    GaussianParams gp;
    gp.mu = gaussIter.value()["mu"].get<double>();
    const double sigma = gaussIter.value()["sigma"].get<double>();
    gp.tau = 1. / (sigma * sigma);
    gauss[gaussIter.key()] = gp;
  }
  prob.readJson (j["prob"]);
  rate.readJson (j["rate"]);
}

ParamAssign GaussianModelParams::eventProbs (double traceRate) const {
  ParamAssign e;
  for (const auto& param_rate: rate.defs) {
    const double waitProb = exp (-traceRate * param_rate.second.get<double>());
    e.defs[waitEventFuncName(param_rate.first)] = waitProb;
    e.defs[exitEventFuncName(param_rate.first)] = 1. - waitProb;
  }
  return e;
}

ParamAssign GaussianModelParams::params (double traceRate) const {
  return ParamAssign (prob.combine (eventProbs (traceRate)));
}
