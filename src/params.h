#ifndef PARAMS_INCLUDED
#define PARAMS_INCLUDED

#include <map>
#include <string>
#include "weight.h"
#include "jsonio.h"

using namespace std;
using json = nlohmann::json;

struct ParamsBase {
  ParamDefs defs;

  void readJson (const json& json);
  void writeJson (ostream& out) const;
};
typedef JsonLoader<ParamsBase> Params;

#endif /* PARAMS_INCLUDED */
