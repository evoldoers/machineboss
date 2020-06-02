#ifndef WEIGHT_INCLUDED
#define WEIGHT_INCLUDED

#include <list>
#include <set>
#include "json.hpp"

#define WeightMacroSymbolPlaceholder       "%"
#define WeightMacroAlphabetSizePlaceholder "#"
#define WeightMacroDefaultMacro            "$p" WeightMacroSymbolPlaceholder ""
#define WeightMacroUniformPriorMacro       "1/" WeightMacroAlphabetSizePlaceholder ""

namespace MachineBoss {

using namespace std;
using json = nlohmann::json;

typedef struct ExprStruct const* ExprPtr;
typedef size_t ExprIndex;
typedef list<ExprStruct>::const_iterator ExprIter;

struct BinaryExprArgs {
  ExprPtr l, r;
};

union ExprArgs {
  int intValue;
  double doubleValue;
  string* param;
  ExprPtr arg;
  BinaryExprArgs binary;
};

enum ExprType {
  Mul, Add, Sub, Div,  // binary
  Pow, Log, Exp,  // unary
  Dbl, Int,  // constants
  Param,  // parameter
  Null
};
struct ExprStruct {
  ExprType type;
  ExprArgs args;
  ExprIndex index;
  ExprStruct() : type(Null) { }
};

typedef ExprPtr WeightExpr;
typedef map<string,WeightExpr> ParamDefs;

typedef vector<size_t> ExprRefCounts;
typedef map<WeightExpr,string> ExprMemos;

struct WeightAlgebra {
  static WeightExpr zero();
  static WeightExpr one();

  static WeightExpr intConstant (int value);
  static WeightExpr doubleConstant (double value);

  static WeightExpr param (const string& name);

  static WeightExpr multiply (const WeightExpr& l, const WeightExpr& r);  // l*r
  static WeightExpr add (const WeightExpr& l, const WeightExpr& r);  // l+r
  static WeightExpr subtract (const WeightExpr& l, const WeightExpr& r);  // l-r
  static WeightExpr divide (const WeightExpr& l, const WeightExpr& r);  // l/r
  static WeightExpr power (const WeightExpr& a, const WeightExpr& b);  // a^b
  static WeightExpr logOf (const WeightExpr& p);  // log(p)
  static WeightExpr expOf (const WeightExpr& p);  // exp(p)

  static WeightExpr minus (const WeightExpr& x);  // 0 - x
  static WeightExpr negate (const WeightExpr& p);  // 1 - p
  static WeightExpr reciprocal (const WeightExpr& p);  // 1 / p
  static WeightExpr geometricSum (const WeightExpr& p);  // 1 / (1 - p)

  static bool isZero (const json& w);
  static bool isOne (const json& w);

  static bool isZero (const WeightExpr& w);
  static bool isOne (const WeightExpr& w);

  static bool isNumber (const WeightExpr& w);
  static double asDouble (const WeightExpr& w);

  static WeightExpr bind (const WeightExpr& w, const ParamDefs& defs);
  
  static double eval (const WeightExpr& w, const ParamDefs& defs, const set<string>* excludedDefs = NULL);

  static WeightExpr deriv (const WeightExpr& w, const ParamDefs& defs, const string& param);
  static set<string> params (const WeightExpr& w, const ParamDefs& defs);
  static vector<string> toposortParams (const ParamDefs& defs);
  
  static string toString (const WeightExpr& w, const ParamDefs& defs, int parentPrecedence = 0);

  static ParamDefs exclude (const ParamDefs& defs, const string& param);

  static void toJsonStream (ostream&, const WeightExpr& w, const ExprMemos* memos = NULL);
  static void toJsonStream (ostream&, const ParamDefs& defs, const ExprMemos* memos = NULL);

  static string toJsonString (const ParamDefs& defs, const ExprMemos* memos = NULL);
  static string toJsonString (const WeightExpr& w, const ExprMemos* memos = NULL);

  static json toJson (const ParamDefs& defs, const ExprMemos* memos = NULL);
  static json toJson (const WeightExpr& w, const ExprMemos* memos = NULL);

  static WeightExpr fromJson (const json& j, const ParamDefs* defs = NULL);

  static map<string,WeightExpr> makeSymbolExprs (const vector<string>& alphabet, const string& macro = string(WeightMacroDefaultMacro));
  
  // trace refcount of functions. also used by params()
  static ExprRefCounts zeroRefCounts();
  static void countRefs (const WeightExpr w, ExprRefCounts& counts, set<string>& params, const ParamDefs& defs, const WeightExpr parent = NULL);
  static ExprIter exprBegin();
};

}  // end namespace

#endif /* WEIGHT_INCLUDED */
