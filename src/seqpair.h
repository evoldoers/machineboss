#ifndef SEQPAIR_INCLUDED
#define SEQPAIR_INCLUDED

#include <string>
#include <json.hpp>
#include "vguard.h"

using namespace std;
using json = nlohmann::json;

struct NamedSeq {
  string name;
  vguard<string> seq;
  void readJson (const json& json);
  void writeJson (ostream& out) const;
};

struct SeqPair {
  NamedSeq input, output;
  void readJson (istream& in);
  void readJson (const json& json);
  void writeJson (ostream& out) const;
  static SeqPair fromJson (istream& in);
  static SeqPair fromFile (const char* filename);
};

#endif /* SEQPAIR_INCLUDED */
