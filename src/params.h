#ifndef PARAMS_INCLUDED
#define PARAMS_INCLUDED

#include <map>
#include <string>
#include <json.hpp>
#include "vguard.h"

using namespace std;
using json = nlohmann::json;

struct Params {
  map<string,double> param;

  void readJson (istream& in);
  void readJson (const json& json);
  void writeJson (ostream& out) const;
  static Params fromJson (istream& in);
  static Params fromFile (const char* filename);
};

#endif /* PARAMS_INCLUDED */
