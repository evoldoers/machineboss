#ifndef TRACE_INCLUDED
#define TRACE_INCLUDED

#include <list>
#include <iostream>

#include "../../ext/nlohmann_json/json.hpp"
#include "../vguard.h"

using namespace std;
using json = nlohmann::json;

typedef long TraceIndex;

struct Trace {
  vguard<double> sample;
  
  void writeJson (ostream& out) const;
  void readJson (const json&);

  void readText (istream& in);
  void readFast5 (const string& filename);
};

struct TraceList {
  list<Trace> trace;
  list<string> filename;
  
  inline size_t size() const { return trace.size(); }
  void readJson (const string& filename);
  void readText (const string& filename);
  void readFast5 (const string& filename);
};

struct TraceParams {
  double scale, shift;

  TraceParams();  // defaults: scale = 1, shift = 0
  
  void writeJson (ostream& out) const;
  void readJson (const json&);
};

struct TraceListParams {
  vguard<TraceParams> params;

  void init (const TraceList&);
  inline size_t size() const { return params.size(); }
  
  void writeJson (ostream& out) const;
  void readJson (const json&);
};

#endif /* TRACE_INCLUDED */
