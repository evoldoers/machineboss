#include <fstream>
#include "params.h"
#include "schema.h"
#include "util.h"

void ParamsBase::readJson (const json& pj) {
  MachineSchema::validateOrDie ("params", pj);
  for (auto iter = pj.begin(); iter != pj.end(); ++iter)
    defs[iter.key()] = iter.value();
}

void ParamsBase::writeJson (ostream& out) const {
  out << "{";
  size_t n = 0;
  for (auto& pv: defs)
    out << (n++ ? "," : "") << "\"" << pv.first << "\":" << pv.second;
  out << "}";
}
