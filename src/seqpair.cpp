#include <fstream>
#include "seqpair.h"
#include "schema.h"
#include "util.h"

void NamedSeq::readJson (const json& json) {
  if (json.count("name"))
    name = json.at("name").get<string>();
  seq.clear();
  for (const auto& js: json.at("sequence"))
    seq.push_back (js.get<string>());
}

void NamedSeq::writeJson (ostream& out) const {
  out << "{\"name\":\"" << name << "\",\"sequence\":[";
  for (size_t n = 0; n < seq.size(); ++n)
    out << (n > 0 ? "," : "") << "\"" << seq[n] << "\"";
  out << "]}";
}

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
  out << "}";
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
