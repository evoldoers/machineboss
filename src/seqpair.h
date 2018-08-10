#ifndef SEQPAIR_INCLUDED
#define SEQPAIR_INCLUDED

#include <string>
#include "jsonio.h"
#include "machine.h"
#include "schema.h"
#include "vguard.h"

using namespace std;
using json = nlohmann::json;

template<typename Symbol>
struct NamedSeq {
  string name;
  vguard<Symbol> seq;
  void readJson (const json& json) {
    MachineSchema::validateOrDie ("namedsequence", json);
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

typedef NamedSeq<InputSymbol> NamedInputSeq;
typedef NamedSeq<OutputSymbol> NamedOutputSeq;

struct SeqPair {
  NamedInputSeq input;
  NamedOutputSeq output;
  void readJson (const json&);
  void writeJson (ostream&) const;
};

struct Envelope {
  typedef long InputIndex;
  typedef long OutputIndex;

  InputIndex inLen;
  OutputIndex outLen;
  vguard<InputIndex> inStart, inEnd;

  inline bool contains (InputIndex x, OutputIndex y) const {
    return y >= 0 && y <= outLen && x >= inStart[y] && x < inEnd[y];
  }

  vguard<long long> offsets() const;  // offsets[n] = sum_{i=0}^n (inEnd[i] - inStart[i])
  Envelope (const SeqPair& sp);
};

struct SeqPairList {
  list<SeqPair> seqPairs;
  list<Envelope> fullEnvelopes() const;
  void readJson (const json&);
  void writeJson (ostream&) const;
};

#endif /* SEQPAIR_INCLUDED */
