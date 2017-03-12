#ifndef WEIGHT_INCLUDED
#define WEIGHT_INCLUDED

#include <set>
#include <json.hpp>

using namespace std;
using json = nlohmann::json;

typedef json WeightExpr;
typedef map<string,json> ParamDefs;

struct WeightAlgebra {
  static WeightExpr multiply (const WeightExpr& l, const WeightExpr& r);  // l*r
  static WeightExpr add (const WeightExpr& l, const WeightExpr& r);  // l+r
  static WeightExpr subtract (const WeightExpr& l, const WeightExpr& r);  // l-r
  static WeightExpr divide (const WeightExpr& l, const WeightExpr& r);  // l/r
  static WeightExpr power (const WeightExpr& a, const WeightExpr& b);  // a^b
  static WeightExpr logOf (const WeightExpr& p);  // log(p)
  static WeightExpr expOf (const WeightExpr& p);  // exp(p)

  static WeightExpr negate (const WeightExpr& p);
  static WeightExpr reciprocal (const WeightExpr& p);
  static WeightExpr geometricSum (const WeightExpr& p);

  static bool isZero (const WeightExpr& w);
  static bool isOne (const WeightExpr& w);

  static string opcode (const WeightExpr& w);
  static const json& operands (const WeightExpr& w);

  static double eval (const WeightExpr& w, const ParamDefs& defs);
  static WeightExpr deriv (const WeightExpr& w, const ParamDefs& defs, const string& param);
  static set<string> params (const WeightExpr& w, const ParamDefs& defs);
  static string toString (const WeightExpr& w, const ParamDefs& defs);

  static ParamDefs exclude (const ParamDefs& defs, const string& param);
};

#endif /* WEIGHT_INCLUDED */
