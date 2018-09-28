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
  void readJsonWithDefaultSeq (const json& j, const vguard<Symbol>& defaultSeq) {
    if (j.count("name"))
      name = j.at("name").get<string>();
    if (j.count("sequence")) {
      seq.clear();
      for (const auto& js: j.at("sequence"))
	seq.push_back (js.get<Symbol>());
      Require (seq.size() == defaultSeq.size()
	       && mismatch (seq.begin(), seq.end(), defaultSeq.begin()).first == seq.end(),
	       "Sequence pair mismatch\nSequence: %s\nExpected: %s\n",
	       json(seq).dump().c_str(),
	       json(defaultSeq).dump().c_str());
    } else
      seq = defaultSeq;
  }
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
  typedef pair<InputSymbol,OutputSymbol> AlignCol;
  typedef list<AlignCol> AlignPath;
  NamedInputSeq input;
  NamedOutputSeq output;
  AlignPath alignment;
  void readJson (const json&);
  void writeJson (ostream&) const;
};

struct Envelope {
  typedef long InputIndex;
  typedef long OutputIndex;
  typedef long long Offset;

  InputIndex inLen;
  OutputIndex outLen;
  // (x,y) is contained within the envelope <==> x is in the half-open interval [i,j) where i=inStart[y], j=inEnd[y]
  vguard<InputIndex> inStart, inEnd;

  inline bool contains (InputIndex x, OutputIndex y) const {
    return y >= 0 && y <= outLen && x >= inStart[y] && x < inEnd[y];
  }

  inline static bool overlapping (InputIndex start1, InputIndex end1, InputIndex start2, InputIndex end2) {
    // two input intervals overlap if neither strictly precedes the other
    // in this case the intervals are half-open [start,end) (c.f. STL iterators) so #1 strictly precedes #2 if start2 >= end1
    return !(start1 >= end2 || start2 >= end1);
  }

  vguard<Offset> offsets() const;  // offsets[y] = sum_{k=0}^{y-1} (inEnd[k] - inStart[k])
  bool fits (const SeqPair&) const;
  bool connected() const;

  Envelope();
  Envelope (const SeqPair& sp);   // calls initPath if sp.trans is nonempty, otherwise calls initFull

  void clear();
  void initFull (const SeqPair&);
  void initPath (const SeqPair::AlignPath&);
  void initPathVicinity (const SeqPair::AlignPath&, size_t width);

  void writeJson (ostream&) const;

  static Envelope fullEnvelope (const SeqPair&);
  static Envelope pathEnvelope (const SeqPair::AlignPath&);
};

struct SeqPairList {
  list<SeqPair> seqPairs;
  list<Envelope> envelopes() const;
  void readJson (const json&);
  void writeJson (ostream&) const;
};

#endif /* SEQPAIR_INCLUDED */
