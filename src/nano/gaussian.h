#ifndef GAUSSIAN_INCLUDED
#define GAUSSIAN_INCLUDED

#include <json.hpp>
#include "../machine.h"
#include "../params.h"

using namespace std;
using json = nlohmann::json;

struct EventFuncNamer {
  static string waitEventFuncName (const string& rateParam);
  static string exitEventFuncName (const string& rateParam);
};

struct GaussianParams {
  double mu, tau;  // mean & precision of Gaussian
  GaussianParams();
};

struct GaussianModelParams : EventFuncNamer {
  map<OutputSymbol,GaussianParams> gauss;
  ParamAssign prob, rate;

  ParamAssign eventProbs (double traceRate) const;  // probabilities of wait & exit events
  ParamAssign params (double traceRate) const;  // merges prob & eventProbs
  
  json asJson() const;
  void writeJson (ostream& out) const;
  void readJson (const json& json);
};

#endif /* GAUSSIAN_INCLUDED */
