#include <fstream>
#include "seqpair.h"
#include "schema.h"
#include "util.h"

void SeqPair::readJson (const json& pj) {
  MachineSchema::validateOrDie ("seqpair", pj);
  input.name = "input";
  output.name = "output";
  input.readJson (pj.at("input"));
  output.readJson (pj.at("output"));
}

void SeqPair::writeJson (ostream& out) const {
  out << "{\"input\":";
  input.writeJson (out);
  out << ",\"output\":";
  output.writeJson (out);
  out << "}";
}

Envelope::Envelope (const SeqPair& sp) :
  inLen (sp.input.seq.size()),
  outLen (sp.output.seq.size()),
  inStart (outLen + 1, 0),
  inEnd (outLen + 1, inLen + 1)
{ }

bool Envelope::fits (const SeqPair& sp) const {
  return inLen == sp.input.seq.size() && outLen == sp.output.seq.size();
}

bool Envelope::connected() const {
  bool conn = overlapping (inStart[0], inEnd[0], 0, 1);
  for (OutputIndex y = 1; conn && y <= outLen; ++y)
    conn = conn && overlapping (inStart[y-1], inEnd[y-1], inStart[y], inEnd[y]);
  return conn && overlapping (inStart[outLen], inEnd[outLen], inLen, inLen + 1);
}

vguard<Envelope::Offset> Envelope::offsets() const {
  // offsets[y] = sum_{k=0}^{y-1} (inEnd[k] - inStart[k])
  // where 0 <= y <= outLen
  vguard<Envelope::Offset> result;
  result.reserve (outLen + 2);
  result.push_back (0);
  for (OutputIndex y = 0; y <= outLen; ++y)
    result.push_back (result.back() + inEnd[y] - inStart[y]);
  return result;
}

Envelope Envelope::fullEnvelope (const SeqPair& sp) {
  return Envelope (sp);
}

list<Envelope> SeqPairList::fullEnvelopes() const {
  list<Envelope> envs;
  for (const auto& sp: seqPairs)
    envs.push_back (Envelope (sp));
  return envs;
}

void SeqPairList::readJson (const json& pj) {
  MachineSchema::validateOrDie ("seqpairlist", pj);
  for (const auto& j: pj)
    seqPairs.push_back (JsonLoader<SeqPair>::fromJson(j));
}

void SeqPairList::writeJson (ostream& out) const {
  out << "[";
  size_t n = 0;
  for (const auto& sp: seqPairs) {
    out << (n++ ? ",\n " : "");
    sp.writeJson (out);
  }
  out << "]";
}
