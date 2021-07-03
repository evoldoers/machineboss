#ifndef PARAMS_INCLUDED
#define PARAMS_INCLUDED

#include <map>
#include <string>
#include "weight.h"
#include "jsonio.h"

// Params class extends ParamDefs with some helpers
namespace MachineBoss {

using namespace std;
using json = nlohmann::json;

class Params {
public:
  ParamDefs defs;
  void writeJson (ostream&) const;
  Params combine (const Params& p, bool overwriteOwnDefs = false) const;  // like JavaScript's extend(), param definitions in p will override param defs in this object
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

}  // end namespace

#endif /* PARAMS_INCLUDED */
