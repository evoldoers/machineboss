#include <fstream>
#include "params.h"
#include "schema.h"
#include "util.h"

void Params::writeJson (ostream& out) const {
  out << WeightAlgebra::toJsonString (defs);
}

void Params::readJsonWithSchema (const json& pj, const char* schemaName) {
  MachineSchema::validateOrDie (schemaName, pj);
  for (auto iter = pj.begin(); iter != pj.end(); ++iter)
    defs[iter.key()] = WeightAlgebra::fromJson (iter.value());
  (void) WeightAlgebra::toposortParams (defs);  // check for cyclic dependencies
}

Params Params::combine (const Params& p, bool overwriteOwnDefs) const {
  Params c (*this);
  // check for consistency as we go
  for (auto it = p.defs.begin(); it != p.defs.end(); ++it) {
    const string& name = it->first;
    const WeightExpr def = it->second;
    if (!overwriteOwnDefs && c.defs.count(name)) {
      const string cDefStr = WeightAlgebra::toJsonString (c.defs.at (name));
      const string pDefStr = WeightAlgebra::toJsonString (def);
      Require (cDefStr == pDefStr, "Inconsistent parameter definitions for %s: %s vs %s", name.c_str(), cDefStr.c_str(), pDefStr.c_str());
    } else
      c.defs[name] = def;
  }
  return c;
}

void Params::clear() {
  defs.clear();
}

void ParamAssign::readJson (const json& pj) {
  readJsonWithSchema (pj, "params");
}

void ParamFuncs::readJson (const json& pj) {
  readJsonWithSchema (pj, "defs");
}
