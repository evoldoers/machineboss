#include <fstream>
#include "seqpair.h"
#include "schema.h"
#include "util.h"

void SeqPair::readJson (istream& in) {
  json pj;
  in >> pj;
  readJson(pj);
}

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
  out << "}" << endl;
}

SeqPair SeqPair::fromJson (istream& in) {
  SeqPair sp;
  sp.readJson (in);
  return sp;
}

SeqPair SeqPair::fromFile (const char* filename) {
  ifstream infile (filename);
  if (!infile)
    Fail ("File not found: %s", filename);
  return fromJson (infile);
}
