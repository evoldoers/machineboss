#ifndef WEIGHT_INCLUDED
#define WEIGHT_INCLUDED

#include <json.hpp>
#include "params.h"

using namespace std;
using json = nlohmann::json;

typedef json TransWeight;

struct WeightAlgebra {
  static TransWeight multiply (const TransWeight& l, const TransWeight& r);  // l*r
  static TransWeight add (const TransWeight& l, const TransWeight& r);  // l+r
  static TransWeight subtract (const TransWeight& l, const TransWeight& r);  // l-r
  static TransWeight divide (const TransWeight& l, const TransWeight& r);  // l/r
  static TransWeight pow (const TransWeight& a, const TransWeight& b);  // a^b
  static TransWeight log (const TransWeight& p);  // log(p)
  static TransWeight exp (const TransWeight& p);  // exp(p)
  static TransWeight geometricSum (const TransWeight& p);

  static string opcode (const TransWeight& w);
  static const json& operands (const TransWeight& w);

  static double eval (const TransWeight& w, const Params& params);
  static TransWeight deriv (const TransWeight& w, const string& param);
};

#endif /* WEIGHT_INCLUDED */
