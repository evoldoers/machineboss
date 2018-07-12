#ifndef PARAMS_INCLUDED
#define PARAMS_INCLUDED

#include <map>
#include <string>
#include "weight.h"
#include "jsonio.h"

using namespace std;
using json = nlohmann::json;

class Params {
public:
  ParamDefs defs;
  void writeJson (ostream&) const;
  Params combine (const Params&) const;
  void clear();
protected:
  void readJsonWithSchema (const json&, const char* schemaName);
};

struct ParamAssign : Params {
  ParamAssign() { }
  ParamAssign (const Params& p) : Params(p) { }
  void readJson (const json& json);
};

struct ParamFuncs : Params {
  ParamFuncs() { }
  ParamFuncs (const Params& p) : Params(p) { }
  void readJson (const json& json);
};

#endif /* PARAMS_INCLUDED */
