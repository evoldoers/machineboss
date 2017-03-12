#include <fstream>
#include "constraints.h"
#include "schema.h"
#include "util.h"

void ConstraintsBase::readJson (const json& pj) {
  MachineSchema::validateOrDie ("constraints", pj);
  norm.clear();
  for (const auto& n: pj.at("norm")) {
    vguard<string> cons;
    for (const auto& p: n)
      cons.push_back (p.get<string>());
    norm.push_back (cons);
  }
}

void ConstraintsBase::writeJson (ostream& out) const {
  out << "{\"norm\":[";
  size_t n = 0;
  for (auto& c: norm) {
    out << (n++ ? "," : "") << "[";
    size_t nj = 0;
    for (auto& p: c)
      out << (nj++ ? "," : "") << "\"" << p << "\"";
    out << "]";
  }
  out << "]}" << endl;
}

Params ConstraintsBase::defaultParams() const {
  Params params;
  for (auto& c: norm)
    for (auto& cp: c)
      params.defs[cp] = 1. / (double) c.size();
  return params;
}
