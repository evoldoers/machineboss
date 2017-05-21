#include <cassert>
#include <iostream>
#include <string>

// comment out if no HDF5
#include "../../ext/fast5/fast5.hpp"

#include "../jsonio.h"
#include "../regexmacros.h"
#include "trace.h"

void Trace::normalize() {
  const double m0 = sample.size();
  double m1 = 0, m2 = 0;
  for (auto x: sample) {
    m1 += x;
    m2 += x*x;
  }
  const double mean = m1 / m0, sd = sqrt(m2/m0 - mean*mean);
  for (auto& x: sample)
    x = (x - mean) / sd;
}

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

  // comment out the rest of this function if no HDF5 libraries are installed
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
}

TraceParams::TraceParams() :
  scale(1),
  shift(0),
  rate(1)
{ }

void TraceParams::writeJson (ostream& out) const {
  out << "{\"name\":\"" << escaped_str(name)
      << "\",\"scale\":" << scale
      << ",\"shift\":" << shift
      << ",\"rate\":" << rate << "}";
}

void TraceParams::readJson (const json& j) {
  name = j.at("name").get<string>();
  scale = j.at("scale").get<double>();
  shift = j.at("shift").get<double>();
  rate = j.at("rate").get<double>();
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
