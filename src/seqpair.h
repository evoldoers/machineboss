#ifndef SEQPAIR_INCLUDED
#define SEQPAIR_INCLUDED

#include <string>
#include <json.hpp>
#include "trans.h"
#include "vguard.h"

using namespace std;
using json = nlohmann::json;

template<typename Symbol>
struct NamedSeq {
  string name;
  vguard<Symbol> seq;
  void readJson (const json& json) {
    if (json.count("name"))
      name = json.at("name").get<string>();
    seq.clear();
    for (const auto& js: json.at("sequence"))
      seq.push_back (js.get<Symbol>());
  }
  void writeJson (ostream& out) const {
    out << "{\"name\":\"" << name << "\",\"sequence\":[";
    for (size_t n = 0; n < seq.size(); ++n)
      out << (n > 0 ? "," : "") << "\"" << seq[n] << "\"";
    out << "]}";
  }
};

struct SeqPair {
  NamedSeq<InputSymbol> input;
  NamedSeq<OutputSymbol> output;
  void readJson (istream& in);
  void readJson (const json& json);
  void writeJson (ostream& out) const;
  static SeqPair fromJson (istream& in);
  static SeqPair fromFile (const char* filename);
};

#endif /* SEQPAIR_INCLUDED */
