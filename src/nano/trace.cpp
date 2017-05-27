#include <cassert>
#include <iostream>
#include <string>

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
  const regex num_re (RE_WHITE_OR_EMPTY "-?" RE_NUMERIC_CHAR_CLASS RE_DOT_STAR);
  string line;
  while (getline(in,line))
    if (regex_match (line, num_re))
      sample.push_back (stod (line));
    else
      cerr << "Skipping line: " << line << endl;
}

void Trace::readFast5 (const string& filename, const string& groupName, const string& readName) {
  name = filename;
  sample.clear();

  fast5::File f;
  f.open(filename);
  if (f.have_raw_samples()) {
    const auto raw = f.get_raw_samples (readName);
    const auto raw_params = f.get_raw_samples_params (readName);
    const auto event_detection_params = f.get_eventdetection_events_params (groupName, readName);
    const size_t eventStart = event_detection_params.start_time - raw_params.start_time;
    sample.insert (sample.end(), raw.begin() + eventStart, raw.end());
  }
}

void TraceList::readJson (const string& fn) {
  trace.push_back (JsonReader<Trace>::fromFile (fn));
  trace.back().name = fn;
}

void TraceList::readText (const string& fn) {
  ifstream in (fn);
  if (!in)
    Fail ("File not found: %s", fn.c_str());
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
