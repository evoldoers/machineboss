#ifndef CONSTRAINTS_INCLUDED
#define CONSTRAINTS_INCLUDED

#include <map>
#include <string>
#include "jsonio.h"
#include "params.h"
#include "vguard.h"

using namespace std;
using json = nlohmann::json;

struct ConstraintsBase {
  vguard<vguard<string> > norm;

  void readJson (const json& json);
  void writeJson (ostream& out) const;

  Params defaultParams() const;
};
typedef JsonLoader<ConstraintsBase> Constraints;

#endif /* CONSTRAINTS_INCLUDED */
