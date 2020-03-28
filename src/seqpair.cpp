#include <fstream>
#include "seqpair.h"
#include "schema.h"
#include "util.h"

using namespace MachineBoss;

void SeqPair::readJson (const json& pj) {
  MachineSchema::validateOrDie ("seqpair", pj);
  input.name = "input";
  output.name = "output";
  if (pj.count("alignment")) {
    vguard<InputSymbol> in;
    vguard<OutputSymbol> out;
    for (const auto& col: pj.at("alignment")) {
      const InputSymbol inSym = col[0];
      const OutputSymbol outSym = col[1];
      if (inSym.size())
	in.push_back (inSym);
      if (outSym.size())
	out.push_back (outSym);
      alignment.push_back (AlignCol (inSym, outSym));
    }
    if (pj.count("input"))
      input.readJsonWithDefaultSeq (pj.at("input"), in);
    else
      input.seq = in;
    if (pj.count("output"))
      output.readJsonWithDefaultSeq (pj.at("output"), out);
    else
      output.seq = out;
    if (pj.count("meta"))
      metadata = pj.at("meta");
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
    for (const auto& col: alignment)
      out << (n++ ? "," : "")
	  << "[\"" << escaped_str(col.first)
	  << "\",\"" << escaped_str(col.second) << "\"]";
    out << "]";
  }
  if (!metadata.is_null())
    out << ",\"meta\":" << metadata;
  out << "}";
}

SeqPair::AlignPath SeqPair::getAlignment (const MachinePath& mp) {
  AlignPath ap;
  for (const auto& t: mp.trans)
    if (!t.isSilent())
      ap.push_back (AlignCol (t.in, t.out));
  return ap;
}

vguard<InputSymbol> SeqPair::getInput (const AlignPath& ap) {
  vguard<InputSymbol> in;
  for (const auto& col: ap)
    if (col.first.size())
      in.push_back (col.first);
  return in;
}

vguard<OutputSymbol> SeqPair::getOutput (const AlignPath& ap) {
  vguard<OutputSymbol> out;
  for (const auto& col: ap)
    if (col.second.size())
      out.push_back (col.second);
  return out;
}

SeqPair SeqPair::seqPairFromPath (const MachineBoundPath& mp, const char* inputName, const char* outputName) {
  const auto alignment = getAlignment (mp);
  return SeqPair ({ NamedInputSeq ({ inputName, getInput (alignment) }),
	NamedOutputSeq ({ outputName, getOutput (alignment) }),
	alignment,
	json::object ({ { "path", JsonWriter<MachineBoundPath>::toJson (mp) } }) });
}

Envelope::Envelope() {
  clear();
}

Envelope::Envelope (const SeqPair& sp) {
  if (sp.alignment.size())
    initPath (sp.alignment);
  else
    initFull (sp);
  Assert (fits(sp), "Envelope/sequence mismatch");
}

Envelope::Envelope (const SeqPair& sp, size_t width) {
  if (sp.alignment.size())
    initPathArea (sp.alignment, width);
  else
    initFull (sp);
  Assert (fits(sp), "Envelope/sequence mismatch");
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

void Envelope::initPathArea (const SeqPair::AlignPath& cols, size_t width) {
  clear();
  vguard<InputIndex> match;
  vguard<size_t> nBefore;
  nBefore.push_back (0);
  for (const auto& t: cols) {
    const bool gotInput = t.first.size(), gotOutput = t.second.size();
    if (gotInput && gotOutput)
      match.push_back (inLen);
    if (gotInput)
      ++inLen;
    if (gotOutput) {
      ++outLen;
      nBefore.push_back (match.size());
    }
  }
  inStart.clear();
  inEnd.clear();
  for (OutputIndex j = 0; j <= outLen; ++j) {
    InputIndex iStart = 0, iEnd = inLen + 1;
    if (nBefore[j] > width)
      iStart = match[nBefore[j] - width - 1] + 1;
    const size_t nAfter = match.size() - nBefore[j];
    if (nAfter > width)
      iEnd = match[nBefore[j] + width] + 1;
    inStart.push_back (iStart);
    inEnd.push_back (iEnd);
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

Envelope Envelope::pathAreaEnvelope (const SeqPair::AlignPath& path, size_t width) {
  Envelope env;
  env.initPathArea (path, width);
  return env;
}

void Envelope::writeJson (ostream& out) const {
  out << "[";
  for (OutputIndex j = 0; j <= outLen; ++j)
    out << (j ? "," : "") << "[" << inStart[j] << "," << inEnd[j] << "]";
  out << "]";
}

list<Envelope> SeqPairList::envelopes() const {
  list<Envelope> envs;
  for (const auto& sp: seqPairs)
    envs.push_back (Envelope (sp));
  return envs;
}

list<Envelope> SeqPairList::envelopes (size_t width) const {
  list<Envelope> envs;
  for (const auto& sp: seqPairs)
    envs.push_back (Envelope (sp, width));
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
