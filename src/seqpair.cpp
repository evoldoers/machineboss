#include <fstream>
#include "seqpair.h"
#include "schema.h"
#include "util.h"

void SeqPairBase::readJson (const json& pj) {
  MachineSchema::validateOrDie ("seqpair", pj);
  input.name = "input";
  output.name = "output";
  input.readJson (pj.at("input"));
  output.readJson (pj.at("output"));
}

void SeqPairBase::writeJson (ostream& out) const {
  out << "{\"input\":";
  input.writeJson (out);
  out << ",\"output\":";
  output.writeJson (out);
  out << "}";
}

void SeqPairListBase::readJson (const json& pj) {
  MachineSchema::validateOrDie ("seqpairlist", pj);
  for (const auto& j: pj)
    seqPairs.push_back (SeqPair::fromJson(j));
}

void SeqPairListBase::writeJson (ostream& out) const {
  out << "[";
  size_t n = 0;
  for (const auto& sp: seqPairs) {
    out << (n++ ? ",\n " : "");
    sp.writeJson (out);
  }
  out << "]";
}
