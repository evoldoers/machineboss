#include "gaussian.h"

GaussianParams::GaussianParams() :
  mu (0),
  tau (1)
{ }

json GaussianModelParams::asJson() const {
  json jg = json::object()
  for (auto& g: gaussian)
    jg[g.first] = json ({ {"mu", g.second.mu}, {"tau", g.second.tau} });
  return jg;
}

void GaussianModelParams::toJson (ostream& out) const {
  out << asJson();
}

void GaussianModelParams::readJson (const json& jg) {
  for (json::iterator gaussIter = jg.begin(); gaussIter != jg.end(); ++gaussIter) {
    GaussianParams gp;
    gp.mu = gaussIter.value()["mu"].get<double>();
    gp.tau = gaussIter.value()["tau"].get<double>();
    gauss[gaussIter.key().get<string>()] = gp;
  }
}
