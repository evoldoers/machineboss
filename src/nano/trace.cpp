#include <cassert>
#include <iostream>
#include <string>
#include "../../ext/fast5/fast5.hpp"
#include "../jsonio.h"
#include "../regexmacros.h"
#include "trace.h"

void Trace::writeJson (ostream& out) const {
  out << "[" << to_string_join(sample,",") << "]";
}

void Trace::readJson (const json& j) {
  sample.clear();
  sample.reserve (j.size());
  for (const auto& elt: j)
    sample.push_back (elt.get<double>());
}

void Trace::readText (istream& in) {
  sample.clear();
  const regex num_re (RE_WHITE_OR_EMPTY RE_NUMERIC_CHAR_CLASS RE_DOT_STAR);
  string line;
  while (getline(in,line))
    if (regex_match (line, num_re))
      sample.push_back (stod (line));
}

void Trace::readFast5 (const string& filename) {
  name = filename;
  sample.clear();
  fast5::File f;
  f.open(filename);
  if (f.have_raw_samples()) {
    const auto raw = f.get_raw_samples();
    sample.insert (sample.end(), raw.begin(), raw.end());
  }
}

void TraceList::readJson (const string& fn) {
  trace.push_back (JsonReader<Trace>::fromFile (fn));
  trace.back().name = fn;
}

void TraceList::readText (const string& fn) {
  ifstream in (fn);
  Trace t;
  t.name = fn;
  t.readText (in);
  trace.push_back (t);
}

void TraceList::readFast5 (const string& fn) {
  Trace t;
  t.readFast5 (fn);
  trace.push_back (t);
  filename.push_back (fn);
}

TraceParams::TraceParams() :
  scale(1),
  shift(0)
{ }

void TraceParams::writeJson (ostream& out) const {
  out << "{\"scale\":" << scale << ",\"shift\":" << shift << "}";
}

void TraceParams::readJson (const json& j) {
  scale = j.at("scale").get<double>();
  shift = j.at("shift").get<double>();
}

void TraceListParams::init (const TraceList& ts) {
  params = vguard<TraceParams> (ts.size());
}

void TraceListParams::writeJson (ostream& out) const {
  out << "[";
  size_t n = 0;
  for (const auto& p: params)
    out << (n++ ? ",\n " : "")
	<< JsonWriter<TraceParams>::toJsonString(p);
  out << "]\n";
}

void TraceListParams::readJson (const json& j) {
  params.clear();
  params.reserve (j.size());
  for (auto iter = j.begin(); iter != j.end(); ++iter)
    params.push_back (JsonReader<TraceParams>::fromJson (iter.value()));
}
