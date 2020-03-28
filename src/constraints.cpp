#include <fstream>
#include "constraints.h"
#include "schema.h"
#include "util.h"

using namespace MachineBoss;

void Constraints::readJson (const json& pj) {
  MachineSchema::validateOrDie ("constraints", pj);
  if (pj.count("norm"))
    for (const auto& n: pj.at("norm")) {
      vguard<string> cons;
      for (const auto& p: n)
	cons.push_back (p.get<string>());
      norm.push_back (cons);
    }
  if (pj.count("prob"))
    for (const auto& p: pj.at("prob"))
      prob.push_back (p.get<string>());
  if (pj.count("rate"))
    for (const auto& r: pj.at("rate"))
      rate.push_back (r.get<string>());
}

string normConsText (const vguard<string>& c) {
  ostringstream out;
  out << "[";
  size_t nj = 0;
  for (auto& p: c)
    out << (nj++ ? "," : "") << "\"" << escaped_str(p) << "\"";
  out << "]";
  return out.str();
}

void Constraints::writeJson (ostream& out) const {
  out << " {";
  size_t l = 0;
  if (norm.size()) {
    ++l;
    out << "\"norm\":" << endl << "  [";
    size_t nc = 0;
    for (auto& c: norm)
      out << (nc++ ? ",\n   " : "") << normConsText(c);
    out << "]";
  }
  if (prob.size()) {
    out << (l++ ? ",\n  " : "")
	<< "\"prob\":[";
    size_t np = 0;
    for (auto& p: prob)
      out << (np++ ? "," : "") << "\"" << escaped_str(p) << "\"";
    out << "]";
  }
  if (rate.size()) {
    out << (l++ ? ",\n  " : "")
	<< "\"rate\":[";
    size_t nr = 0;
    for (auto& r: rate)
      out << (nr++ ? "," : "") << "\"" << escaped_str(r) << "\"";
    out << "]";
  }
  out << "}" << endl;
}

Params Constraints::defaultParams() const {
  Params params;
  for (auto& c: norm)
    for (auto& cp: c)
      params.defs[cp] = WeightAlgebra::doubleConstant (1. / (double) c.size());
  for (auto& pp: prob)
    params.defs[pp] = WeightAlgebra::doubleConstant (.5);
  for (auto& rp: rate)
    params.defs[rp] = WeightAlgebra::intConstant (1);
  return params;
}

string probType (const string& p) { return string("prob[") + p + "]"; }
string rateType (const string& r) { return string("rate[") + r + "]"; }
string normType (const vguard<string>& c) { return string("norm") + normConsText(c); }
bool checkRedundant (const map<string,string>& type, const string& p, const string& t) {
  const auto count = type.count(p);
  Require (!count || type.at(p) == t,
	  "Inconsistent constraints for %s: %s vs %s", p.c_str(), type.at(p).c_str(), t.c_str());
  return count;
}

map<string,string> Constraints::byParam() const {
  map<string,string> type;
  for (auto& p: prob)
    type[p] = probType(p);
  for (auto& r: rate)
    type[r] = rateType(r);
  for (auto& c: norm) {
    const string ctype = normType(c);
    for (auto& p: c)
      type[p] = ctype;
  }
  return type;
}

Constraints Constraints::combine (const Constraints& cons) const {
  Constraints result (*this);
  // check for consistency as we go
  map<string,string> type = byParam();
  for (auto& p: cons.prob)
    if (!checkRedundant (type, p, probType(p)))
      result.prob.push_back (p);
  for (auto& r: cons.rate)
    if (!checkRedundant (type, r, rateType(r)))
      result.rate.push_back (r);
  for (auto& c: cons.norm) {
    const string ctype = normType(c);
    bool redundant = false;
    for (auto& p: c)
      redundant = checkRedundant (type, p, ctype) || redundant;
    if (!redundant)
      result.norm.push_back (c);
  }
  return result;
}

void Constraints::clear() {
  prob.clear();
  norm.clear();
  rate.clear();
}

bool Constraints::empty() const {
  return prob.empty() && norm.empty() && rate.empty();
}
