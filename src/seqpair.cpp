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
  inEnd (outLen + 1, inLen + 2)
{ }

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
