#ifndef WEIGHT_INCLUDED
#define WEIGHT_INCLUDED

#include <json.hpp>
using namespace std;
using json = nlohmann::json;

typedef json TransWeight;
struct WeightAlgebra {
  static TransWeight multiply (const TransWeight& l, const TransWeight& r);
  static TransWeight add (const TransWeight& l, const TransWeight& r);
  static TransWeight geometricSum (const TransWeight& p);
};

#endif /* WEIGHT_INCLUDED */
