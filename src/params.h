#ifndef PARAMS_INCLUDED
#define PARAMS_INCLUDED

#include <map>
#include <string>
#include <json.hpp>
#include "weight.h"

using namespace std;
using json = nlohmann::json;

struct Params {
  ParamDefs defs;

  void readJson (istream& in);
  void readJson (const json& json);
  void writeJson (ostream& out) const;
  static Params fromJson (istream& in);
  static Params fromFile (const char* filename);
  string toJsonString() const;
};

#endif /* PARAMS_INCLUDED */
