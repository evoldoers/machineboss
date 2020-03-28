#ifndef CONSTRAINTS_INCLUDED
#define CONSTRAINTS_INCLUDED

#include <map>
#include <string>
#include "jsonio.h"
#include "params.h"
#include "vguard.h"

namespace MachineBoss {

using namespace std;
using json = nlohmann::json;

struct Constraints {
  vguard<string> prob;
  vguard<string> rate;
  vguard<vguard<string> > norm;

  void clear();
  bool empty() const;
  
  void readJson (const json& json);
  void writeJson (ostream& out) const;

  Params defaultParams() const;

  map<string,string> byParam() const;
  Constraints combine (const Constraints& cons) const;
};

}  // end namespace

#endif /* CONSTRAINTS_INCLUDED */
