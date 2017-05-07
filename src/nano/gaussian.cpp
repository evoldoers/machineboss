#include "gaussian.h"

GaussianParams::GaussianParams() :
  mu (0),
  tau (1)
{ }

json GaussianModelParams::asJson() const {
  json jg = json::object();
  for (auto& g: gauss)
    jg[g.first] = json ({ {"mu", g.second.mu}, {"tau", g.second.tau} });
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
    gp.tau = gaussIter.value()["tau"].get<double>();
    gauss[gaussIter.key()] = gp;
  }
  prob.readJson (j["prob"]);
  rate.readJson (j["rate"]);
}

Params GaussianModelParams::params() const {
  return prob.combine(rate);
}
