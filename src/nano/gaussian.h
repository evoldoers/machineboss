#ifndef GAUSSIAN_INCLUDED
#define GAUSSIAN_INCLUDED

#include <json.hpp>
#include "../machine.h"
#include "../params.h"

using namespace std;
using json = nlohmann::json;

struct GaussianParams {
  double mu, tau;  // mean & precision of Gaussian
  GaussianParams();
};

struct GaussianModelParams {
  map<OutputSymbol,GaussianParams> gauss;
  ParamAssign prob;
  
  json asJson() const;
  void writeJson (ostream& out) const;
  void readJson (const json& json);
};

#endif /* GAUSSIAN_INCLUDED */
