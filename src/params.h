#ifndef PARAMS_INCLUDED
#define PARAMS_INCLUDED

#include <map>
#include <string>
#include "weight.h"
#include "jsonio.h"

using namespace std;
using json = nlohmann::json;

// Params class extends ParamDefs with some helpers
class Params {
public:
  ParamDefs defs;
  void writeJson (ostream&) const;
  Params combine (const Params&) const;
  void clear();
protected:
  void readJsonWithSchema (const json&, const char* schemaName);
};

// ParamAssign may contain only numerical assignments
struct ParamAssign : Params {
  ParamAssign() { }
  ParamAssign (const Params& p) : Params(p) { }
  void readJson (const json& json);
};

// ParamFuncs may contain arbitrary expressions
struct ParamFuncs : Params {
  ParamFuncs() { }
  ParamFuncs (const Params& p) : Params(p) { }
  void readJson (const json& json);
};

#endif /* PARAMS_INCLUDED */
