#include <math.h>
#include <list>
#include "weight.h"
#include "logsumexp.h"
#include "util.h"
#include "logger.h"

// singleton for storing ExprStruct's
class ExprStructFactory {
private:
  list<ExprStruct> exprStructStorage;
  list<string> paramStorage;
  ExprStruct zeroStr, oneStr;
public:
  ExprPtr zero, one;
  ExprStructFactory() {
    zeroStr.type = oneStr.type = Int;
    zeroStr.args.intValue = 0;
    oneStr.args.intValue = 1;
    zero = &zeroStr;
    one = &oneStr;
  }
  ExprPtr newExpr() {
    exprStructStorage.push_front (ExprStruct());
    return &exprStructStorage.front();
  }
  ExprPtr newParam (const string& param) {
    paramStorage.push_front (param);
    ExprPtr e = newExpr();
    e->type = Param;
    e->args.param = &paramStorage.front();
    return e;
  }
  ExprPtr newInt (int val) {
    if (val == 0)
      return zero;
    if (val == 1)
      return one;
    ExprPtr e = newExpr();
    e->type = Int;
    e->args.intValue = val;
    return e;
  }
  ExprPtr newDouble (double val) {
    if (val == 0.)
      return zero;
    if (val == 1.)
      return one;
    ExprPtr e = newExpr();
    e->type = Dbl;
    e->args.doubleValue = val;
    return e;
  }
  ExprPtr newUnary (ExprType type, ExprPtr arg) {
    Assert (arg, "Null argument to unary function");
    ExprPtr e = newExpr();
    e->type = type;
    e->args.arg = arg;
    return e;
  }
  ExprPtr newBinary (ExprType type, ExprPtr l, ExprPtr r) {
    Assert (l && r, "Null argument to binary function");
    ExprPtr e = newExpr();
    e->type = type;
    e->args.binary.l = l;
    e->args.binary.r = r;
    return e;
  }
};
ExprStructFactory factory;

WeightExpr WeightAlgebra::zero() {
  return factory.zero;
}

WeightExpr WeightAlgebra::one() {
  return factory.one;
}

WeightExpr WeightAlgebra::intConstant (int value) {
  return value == 0 ? factory.zero : (value == 1 ? factory.one : factory.newInt (value));
}

WeightExpr WeightAlgebra::doubleConstant (double value) {
  return value == 0. ? factory.zero : (value == 1. ? factory.one : factory.newDouble (value));
}

WeightExpr WeightAlgebra::param (const string& name) {
  return factory.newParam (name);
}

WeightExpr WeightAlgebra::minus (const WeightExpr& x) {
  return factory.newBinary (Sub, factory.zero, x);
}

WeightExpr WeightAlgebra::negate (const WeightExpr& p) {
  return factory.newBinary (Sub, factory.one, p);
}

WeightExpr WeightAlgebra::reciprocal (const WeightExpr& p) {
  return factory.newBinary (Div, factory.one, p);
}

WeightExpr WeightAlgebra::geometricSum (const WeightExpr& p) {
  return WeightAlgebra::reciprocal (WeightAlgebra::negate (p));
}

WeightExpr WeightAlgebra::divide (const WeightExpr& l, const WeightExpr& r) {
  return (isOne(r) || isZero(l)) ? l : factory.newBinary (Div, l, r);
}

WeightExpr WeightAlgebra::subtract (const WeightExpr& l, const WeightExpr& r) {
  return isZero(r) ? l : factory.newBinary (Sub, l, r);
}

WeightExpr WeightAlgebra::power (const WeightExpr& a, const WeightExpr& b) {
  return isOne(b) ? a : (isZero(b) ? factory.one : factory.newBinary (Pow, a, b));
}

WeightExpr WeightAlgebra::logOf (const WeightExpr& p) {
  return isOne(p) ? factory.zero : (p->type == Exp ? p->args.arg : factory.newUnary (Log, p));
}

WeightExpr WeightAlgebra::expOf (const WeightExpr& p) {
  return isZero(p) ? factory.one : (p->type == Log ? p->args.arg : factory.newUnary (Exp, p));
}

WeightExpr WeightAlgebra::multiply (const WeightExpr& l, const WeightExpr& r) {
  WeightExpr w = NULL;
  if (isOne(l))
    w = r;
  else if (isOne(r))
    w = l;
  else if (isZero(l) || isZero(r))
    w = factory.zero;
  else if (l->type == Int && r->type == Int)
    w = factory.newInt (l->args.intValue * r->args.intValue);
  else if (isNumber(l) && isNumber(r))
    w = factory.newDouble (asDouble(l) * asDouble(r));
  else
    w = factory.newBinary (Mul, l, r);
  return w;
}

WeightExpr WeightAlgebra::add (const WeightExpr& l, const WeightExpr& r) {
  WeightExpr w = NULL;
  if (isZero(l))
    w = r;
  else if (isZero(r))
    w = l;
  else if (l->type == Int && r->type == Int)
    w = factory.newInt (l->args.intValue + r->args.intValue);
  else if (isNumber(l) && isNumber(r))
    w = factory.newDouble (asDouble(l) + asDouble(r));
  else
    w = factory.newBinary (Add, l, r);
  return w;
}

bool WeightAlgebra::isZero (const WeightExpr& w) {
  return w == factory.zero || (w->type == Int && w->args.intValue == 0) || (w->type == Dbl && w->args.doubleValue == 0.);
}

bool WeightAlgebra::isOne (const WeightExpr& w) {
  return w == factory.one || (w->type == Int && w->args.intValue == 1) || (w->type == Dbl && w->args.doubleValue == 1.);
}

bool WeightAlgebra::isNumber (const WeightExpr& w) {
  return w->type == Int || w->type == Dbl;
}

double WeightAlgebra::asDouble (const WeightExpr& w) {
  Assert (isNumber(w), "WeightExpr is not numeric");
  return w->type == Int ? ((double) w->args.intValue) : w->args.doubleValue;
}

bool WeightAlgebra::isZero (const json& w) {
  return w.is_null()
    || (w.is_boolean() && !w.get<bool>())
    || (w.is_number_integer() && w.get<int>() == 0)
    || (w.is_number() && w.get<double>() == 0.);
}

bool WeightAlgebra::isOne (const json& w) {
  return (w.is_boolean() && w.get<bool>())
    || (w.is_number_integer() && w.get<int>() == 1)
    || (w.is_number() && w.get<double>() == 1.);
}

WeightExpr WeightAlgebra::bind (const WeightExpr& w, const ParamDefs& defs) {
  const ExprType op = w->type;
  WeightExpr result = NULL;
  switch (op) {
  case Null:
    throw runtime_error("Attempt to bind null expression");
  case Int:
  case Dbl:
    result = w;
    break;
  case Param:
    {
      const string& n (*w->args.param);
      result = defs.count(n) ? bind (defs.at(n), defs) : w;
    }
    break;
  case Log:
  case Exp:
    result = factory.newUnary (op, bind (w->args.arg, defs));
    break;
  default:
    result = factory.newBinary (op, bind (w->args.binary.l, defs), bind (w->args.binary.r, defs));
    break;
  }
  return result;
}

double WeightAlgebra::eval (const WeightExpr& w, const ParamDefs& defs, const set<string>* excludedDefs) {
  const ExprType op = w->type;
  double result;
  switch (op) {
  case Null:
    result = 0;
    break;
  case Int:
  case Dbl:
    result = asDouble(w);
    break;
  case Param:
    {
      const string& n (*w->args.param);
      if (!defs.count(n) || (excludedDefs && excludedDefs->count(n)))
	throw runtime_error(string("Parameter ") + n + (" not defined"));
      // optimize the special case that definition is a numeric assignment
      const auto& val = defs.at(n);
      if (isNumber(val))
	result = asDouble(val);
      else {
	set<string> innerExcludedDefs;
	if (excludedDefs)
	  innerExcludedDefs.insert (excludedDefs->begin(), excludedDefs->end());
	innerExcludedDefs.insert (n);
	result = eval (defs.at(n), defs, &innerExcludedDefs);
      }
    }
    break;
  case Log:
    result = log (eval (w->args.arg, defs, excludedDefs));
    break;
  case Exp:
    result = exp (eval (w->args.arg, defs, excludedDefs));
    break;
  default:
    const double l = eval (w->args.binary.l, defs, excludedDefs);
    const double r = eval (w->args.binary.r, defs, excludedDefs);
    switch (op) {
    case Mul:
      result = l * r;
      break;
    case Div:
      result = l / r;
      break;
    case Add:
      result = l + r;
      break;
    case Sub:
      result = l - r;
      break;
    case Pow:
      result = pow (l, r);
      break;
    default:
      Abort("Unknown opcode");
    }
  }
  return result;
}

WeightExpr WeightAlgebra::deriv (const WeightExpr& w, const ParamDefs& defs, const string& param) {
  WeightExpr d = NULL;
  const ExprType op = w->type;
  switch (op) {
  case Null:
  case Int:
  case Dbl:
    d = factory.zero;
    break;
  case Param:
    {
      const string& n (*w->args.param);
      if (param == n)
	d = factory.one;
      else if (defs.count(n))
	d = deriv (defs.at(n), exclude(defs,n), param);
      else
	d = factory.zero;
    }
    break;
  case Exp:
    d = multiply (deriv (w->args.arg, defs, param), w);  // w = exp(x), w' = x'exp(x)
    break;
  case Log:
    d = divide (deriv (w->args.arg, defs, param), w->args.arg);  // w = log(x), w' = x'/x
    break;
  default:
    const WeightExpr dl = deriv (w->args.binary.l, defs, param);
    const WeightExpr dr = deriv (w->args.binary.r, defs, param);
    switch (op) {
    case Mul:
      d = add (multiply(dl,w->args.binary.r), multiply(w->args.binary.l,dr));  // w = fg, w' = f'g + g'f
      break;
    case Div:
      d = subtract (divide(dl,w->args.binary.r), multiply(dr,divide(w,w->args.binary.r)));  // w = f/g, w' = f'/g - g'f/g^2
      break;
    case Add:
      d = add (dl, dr);  // w = f + g, w' = f' + g'
      break;
    case Sub:
      d = subtract (dl, dr);  // w = f - g, w' = f' - g'
      break;
    case Pow:
      d = multiply (w, add (multiply(dr,logOf(w->args.binary.l)), multiply(dl,divide(w->args.binary.r,w->args.binary.l))));  // w = a^b, w' = a^b (b'*log(a) + a'b/a)
      break;
    default:
      Abort("Unknown opcode", op);
    }
  }
  return d;
}

set<string> WeightAlgebra::params (const WeightExpr& w, const ParamDefs& defs) {
  set<string> p;
  const ExprType op = w->type;
  switch (op) {
  case Null:
  case Int:
  case Dbl:
    // p is empty
    break;
  case Param:
    {
      const string& n (*w->args.param);
      if (defs.count(n))
	p = params (defs.at(n), exclude(defs,n));
      else
	p.insert (n);
    }
    break;
  case Exp:
  case Log:
    p = params (w->args.arg, defs);
    break;
  default:
    const set<string> lParams = params (w->args.binary.l, defs);
    const set<string> rParams = params (w->args.binary.r, defs);
    p.insert (lParams.begin(), lParams.end());
    p.insert (rParams.begin(), rParams.end());
    break;
  }
  return p;
}

string WeightAlgebra::toString (const WeightExpr& w, const ParamDefs& defs, int parentPrecedence) {
  const ExprType op = w->type;
  string result;
  switch (op) {
  case Null: result = string("0"); break;
  case Int: result = to_string (w->args.intValue); break;
  case Dbl: result = to_string (w->args.doubleValue); break;
  case Param:
    {
      const string& n (*w->args.param);
      result = defs.count(n) ? toString(defs.at(n),exclude(defs,n),parentPrecedence) : n;
    }
    break;
  case Log:
  case Exp:
    result = string(op == Log ? "log" : "exp") + "(" + toString(w->args.arg,defs) + ")";
    break;
  case Pow:
    result = string("pow(") + toString(w->args.binary.l,defs) + "," + toString(w->args.binary.r,defs) + ")";
    break;
  default:
    // Precedence rules

    // a*b: rank 2
    // a needs () if it's anything except a multiplication or division [parent rank 2]
    // b needs () if it's anything except a multiplication or division [parent rank 2]

    // a/b: rank 2
    // a needs () if it's anything except a multiplication or division [parent rank 2]
    // b needs () if it's anything except a constant/function [parent rank 3]

    // a-b: rank 1
    // a never needs () [parent rank 0]
    // b needs () if it's anything except a multiplication or division [parent rank 2]

    // a+b: rank 1
    // a never needs () [parent rank 0]
    // b never needs () [parent rank 0]

    int p, l, r;
    string opcode;
    if (op == Mul) { p = l = r = 2; opcode = "*"; }
    else if (op == Div) { p = l = 2; r = 3; opcode = "/"; }
    else if (op == Sub) { p = 1; l = 0; r = 2; opcode = "-"; }
    else if (op == Add) { p = 1; l = r = 0; opcode = "+"; }
    result = string(parentPrecedence > p ? "(" : "")
      + toString(w->args.binary.l,defs,l)
      + opcode
      + toString(w->args.binary.r,defs,r)
      + (parentPrecedence > p ? ")" : "");
    break;
  }
  return result;
}

ParamDefs WeightAlgebra::exclude (const ParamDefs& defs, const string& param) {
  ParamDefs defsCopy (defs);
  defsCopy.erase (param);
  return defsCopy;
}

string WeightAlgebra::toJsonString (const ParamDefs& defs, const ExprMemos* memos) {
  ostringstream out;
  out << toJson (defs, memos);
  return out.str();
}

string WeightAlgebra::toJsonString (const WeightExpr& w, const ExprMemos* memos) {
  ostringstream out;
  out << toJson (w, memos);
  return out.str();
}

json WeightAlgebra::toJson (const ParamDefs& defs, const ExprMemos* memos) {
  json j = json::object();
  for (const auto& def: defs)
    j[def.first] = toJson (def.second, memos);
  return j;
}

json WeightAlgebra::toJson (const WeightExpr& w, const ExprMemos* memos) {
  json result;
  if (memos && memos->count(w))
    result = memos->at(w);
  else {
    const ExprType op = w->type;
    if (isZero(w))
      result = false;
    else if (isOne(w))
      result = true;
    else
      switch (op) {
      case Null:
	// result = null (implicit)
	break;
      case Int:
	result = w->args.intValue;
	break;
      case Dbl:
	result = w->args.doubleValue;
	break;
      case Param:
	result = *w->args.param;
	break;
      case Log:
	result = json::object ({{"log", toJson (w->args.arg, memos)}});
	break;
      case Exp:
	result = json::object ({{"exp", toJson (w->args.arg, memos)}});
	break;
      case Pow:
	result = json::object ({{"pow", json::array ({ toJson (w->args.binary.l, memos), toJson (w->args.binary.r, memos) })}});
	break;
      default:
	string opcode;
	const WeightExpr l = w->args.binary.l, r = w->args.binary.r;
	json jsonArg;
	vguard<WeightExpr> args;
	switch (op) {
	case Mul: opcode = "*"; break;
	case Add: opcode = "+"; break;
	case Div:
	  if (isOne(l) && r->type == Sub && isOne(r->args.binary.l)) {
	    opcode = "geomsum";
	    jsonArg = toJson (r->args.binary.r, memos);
	    break;
	  }
	  opcode = "/";
	  break;
	case Sub:
	  if (isOne(l)) {
	    opcode = "not";
	    jsonArg = toJson (r, memos);
	    break;
	  }
	  opcode = "-";
	  break;
	default: Abort ("Unknown opcode in toJson"); break;
	}
	if (jsonArg.is_null())
	  jsonArg = json::array ({ toJson(l,memos), toJson(r,memos) });
	result = json::object ({{ opcode, jsonArg }});
	break;
      }
  }
  return result;
}

WeightExpr WeightAlgebra::fromJson (const json& w, const ParamDefs* defs) {
 WeightExpr result = NULL;
 if (w.is_null())
   result = WeightExpr();
 else if (w.is_boolean())
   result = w.get<bool>() ? factory.one : factory.zero;
 else if (w.is_number_integer())
   result = factory.newInt (w.get<int>());
 else if (w.is_number())
   result = factory.newDouble (w.get<double>());
 else if (w.is_string()) {
   const string name = w.get<string>();
   result = defs && defs->count(name) ? defs->at(name) : factory.newParam(name);
 } else if (w.is_array())
    Abort ("Unexpected type in WeightExpr: array");
 else {
   if (!w.is_object())
     Abort ("Unexpected type (%d) in WeightExpr", w.type());
   auto iter = w.begin();
   const string opcode = iter.key();
   const json args = iter.value();
   if (opcode == "log")
     result = logOf (fromJson (args, defs));
   else if (opcode == "exp")
     result = expOf (fromJson (args, defs));
   else if (opcode == "not")
     result = negate (fromJson (args, defs));
   else if (opcode == "geomsum")
     result = geometricSum (fromJson (args, defs));
   else if (opcode == "*")
     result = multiply (fromJson (args[0], defs), fromJson (args[1], defs));
   else if (opcode == "/")
     result = divide (fromJson (args[0], defs), fromJson (args[1], defs));
   else if (opcode == "+")
     result = add (fromJson (args[0], defs), fromJson (args[1], defs));
   else if (opcode == "-")
     result = subtract (fromJson (args[0], defs), fromJson (args[1], defs));
   else
     Abort ("Unknown opcode %s in JSON", opcode.c_str());
 }
 return result;
}

void WeightAlgebra::countRefs (const WeightExpr& w, ExprRefCounts& counts, const WeightExpr parent) {
  switch (w->type) {
  case Null:
  case Int:
  case Dbl:
  case Param:
    break;
  case Log:
  case Exp:
    countRefs (w->args.arg, counts, w);
    break;
  default:
    countRefs (w->args.binary.l, counts, w);
    countRefs (w->args.binary.r, counts, w);
    break;
  }

  if (!counts.count(w)) {
    counts[w].order = counts.size();
    counts[w].expr = w;
  }
  counts[w].refs.insert (parent);
}
