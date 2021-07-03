#ifndef GETPARAMS_INCLUDED
#define GETPARAMS_INCLUDED

#include <string>
#include <map>
#include "softplus.h"

namespace MachineBoss {

inline bool getParams (const map<string,double>& params, const vector<string>& names, double* const p) {
  bool ok = true;
  for (size_t n = 0; n < names.size(); ++n) {
    const string& name = names[n];
    if (!params.count (name)) {
      cerr << "Please define parameter: " << name << endl;
      ok = false;
    } else
      p[n] = params.at (name);
  }
  return ok;
}

}  // end namespace

#endif /* GETPARAMS_INCLUDED */
