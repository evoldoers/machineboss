#include <fstream>
#include "params.h"
#include "schema.h"
#include "util.h"

void Params::writeJson (ostream& out) const {
  out << "{";
  size_t n = 0;
  for (auto& pv: defs)
    out << (n++ ? "," : "") << "\"" << pv.first << "\":" << pv.second;
  out << "}";
}

void Params::readJsonWithSchema (const json& pj, const char* schemaName) {
  MachineSchema::validateOrDie (schemaName, pj);
  for (auto iter = pj.begin(); iter != pj.end(); ++iter)
    defs[iter.key()] = iter.value();
}

Params Params::combine (const Params& p) const {
  Params c (*this);
  for (auto it = p.defs.begin(); it != p.defs.end(); ++it)
    c.defs[it->first] = it->second;
  return c;
}

void ParamAssign::readJson (const json& pj) {
  readJsonWithSchema (pj, "params");
}

void ParamFuncs::readJson (const json& pj) {
  readJsonWithSchema (pj, "defs");
}
