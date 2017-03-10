#ifndef CONSTRAINTS_INCLUDED
#define CONSTRAINTS_INCLUDED

#include <map>
#include <string>
#include <json.hpp>
#include "vguard.h"

using namespace std;
using json = nlohmann::json;

struct Constraints {
  vguard<vguard<string> > norm;

  void readJson (istream& in);
  void readJson (const json& json);
  void writeJson (ostream& out) const;
  static Constraints fromJson (istream& in);
  static Constraints fromFile (const char* filename);
};

#endif /* CONSTRAINTS_INCLUDED */
