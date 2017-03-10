#include <fstream>
#include "constraints.h"
#include "schema.h"
#include "util.h"

void Constraints::readJson (istream& in) {
  json pj;
  in >> pj;
  readJson(pj);
}

void Constraints::readJson (const json& pj) {
  MachineSchema::validateOrDie ("constraints", pj);
  norm.clear();
  for (const auto& n: pj.at("norm")) {
    vguard<string> cons;
    for (const auto& p: n)
      cons.push_back (p.get<string>());
    norm.push_back (cons);
  }
}

void Constraints::writeJson (ostream& out) const {
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

Constraints Constraints::fromJson (istream& in) {
  Constraints c;
  c.readJson (in);
  return c;
}

Constraints Constraints::fromFile (const char* filename) {
  ifstream infile (filename);
  if (!infile)
    Fail ("File not found: %s", filename);
  return fromJson (infile);
}
