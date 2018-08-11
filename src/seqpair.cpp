#include <fstream>
#include "seqpair.h"
#include "schema.h"
#include "util.h"

void SeqPair::readJson (const json& pj) {
  MachineSchema::validateOrDie ("seqpair", pj);
  input.name = "input";
  output.name = "output";
  if (pj.count("alignment")) {
    vguard<InputSymbol> in;
    vguard<OutputSymbol> out;
    for (const auto& col: pj.at("alignment")) {
      const bool gotInput = col.count("in"), gotOutput = col.count("out");
      if (gotInput || gotOutput)
	alignment.push_back (AlignCol (gotInput ? col.at("in").get<string>() : string(),
				       gotOutput ? col.at("out").get<string>() : string()));
      if (gotInput)
	in.push_back (alignment.back().first);
      if (gotOutput)
	out.push_back (alignment.back().second);
    }
    if (pj.count("input"))
      input.readJsonWithDefaultSeq (pj.at("input"), in);
    if (pj.count("output"))
      output.readJsonWithDefaultSeq (pj.at("output"), out);
  } else {
    input.readJson (pj.at("input"));
    output.readJson (pj.at("output"));
  }
}

void SeqPair::writeJson (ostream& out) const {
  out << "{\"input\":";
  input.writeJson (out);
  out << ",\"output\":";
  output.writeJson (out);
  if (alignment.size()) {
    out << ",\"alignment\":[";
    size_t n = 0;
    for (const auto& col: alignment) {
      out << (n++ ? "," : "") << "{";
      if (col.first.size())
	out << "\"in\":\"" << escaped_str(col.first) << "\"";
      if (col.second.size())
	out << (col.first.size() ? "," : "")
	    << "\"out\":\"" << escaped_str(col.second) << "\"";
      out << "}";
    }
    out << "]";
  }
  out << "}";
}

Envelope::Envelope() {
  clear();
}

Envelope::Envelope (const SeqPair& sp) {
  if (sp.alignment.size())
    initPath (sp.alignment);
  else
    initFull (sp);
}

void Envelope::clear() {
  inLen = outLen = 0;
  inStart = vguard<InputIndex> (1, 0);
  inEnd = vguard<InputIndex> (1, 1);
}

void Envelope::initFull (const SeqPair& sp) {
  clear();
  inLen = sp.input.seq.size();
  outLen = sp.output.seq.size();
  inStart = vguard<InputIndex> (outLen + 1, 0);
  inEnd = vguard<InputIndex> (outLen + 1, inLen + 1);
}

void Envelope::initPath (const SeqPair::AlignPath& cols) {
  clear();
  for (const auto& t: cols) {
    const bool gotInput = t.first.size(), gotOutput = t.second.size();
    if (!gotInput && gotOutput) {
      inStart.push_back (inEnd.back() - 1);
      inEnd.push_back (inEnd.back());
      ++outLen;
    } else if (gotInput && !gotOutput) {
      ++inEnd.back();
      ++inLen;
    } else if (gotInput && gotOutput) {
      inStart.push_back (inEnd.back());
      inEnd.push_back (inEnd.back() + 1);
      ++inLen;
      ++outLen;
    }
  }
}

bool Envelope::fits (const SeqPair& sp) const {
  return inLen == sp.input.seq.size() && outLen == sp.output.seq.size();
}

bool Envelope::connected() const {
  bool conn = overlapping (inStart[0], inEnd[0], 0, 1);
  for (OutputIndex y = 1; conn && y <= outLen; ++y)
    conn = conn && overlapping (inStart[y-1], inEnd[y-1] + 1, inStart[y], inEnd[y]);  // the +1 in (inEnd[y-1] + 1) allows for diagonal connections
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
  Envelope env;
  env.initFull (sp);
  return env;
}

Envelope Envelope::pathEnvelope (const SeqPair::AlignPath& path) {
  Envelope env;
  env.initPath (path);
  return env;
}

list<Envelope> SeqPairList::envelopes() const {
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
