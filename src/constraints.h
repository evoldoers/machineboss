#ifndef CONSTRAINTS_INCLUDED
#define CONSTRAINTS_INCLUDED

#include <map>
#include <string>
#include "jsonio.h"
#include "params.h"
#include "vguard.h"

using namespace std;
using json = nlohmann::json;

struct Constraints {
  vguard<string> prob;
  vguard<vguard<string> > norm;

  void readJson (const json& json);
  void writeJson (ostream& out) const;

  Params defaultParams() const;
};

#endif /* CONSTRAINTS_INCLUDED */
