#ifndef SEQPAIR_INCLUDED
#define SEQPAIR_INCLUDED

#include <string>
#include "jsonio.h"
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

struct SeqPairBase {
  NamedSeq<InputSymbol> input;
  NamedSeq<OutputSymbol> output;
  void readJson (const json&);
  void writeJson (ostream&) const;
};
typedef JsonLoader<SeqPairBase> SeqPair;

struct SeqPairListBase {
  list<SeqPair> seqPairs;
  void readJson (const json&);
  void writeJson (ostream&) const;
};
typedef JsonLoader<SeqPairListBase> SeqPairList;

#endif /* SEQPAIR_INCLUDED */
