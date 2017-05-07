#include <math.h>
#include "weight.h"
#include "logsumexp.h"
#include "util.h"

WeightExpr WeightAlgebra::negate (const WeightExpr& p) {
  return WeightExpr::object ({{"-", WeightExpr::array ({true, p})}});
}

WeightExpr WeightAlgebra::reciprocal (const WeightExpr& p) {
  return WeightExpr::object ({{"/", WeightExpr::array ({true, p})}});
}

WeightExpr WeightAlgebra::geometricSum (const WeightExpr& p) {
  return WeightAlgebra::reciprocal (WeightAlgebra::negate (p));
}

WeightExpr WeightAlgebra::divide (const WeightExpr& l, const WeightExpr& r) {
  return (isOne(r) || isZero(l)) ? WeightExpr(l) : WeightExpr::object ({{"/", WeightExpr::array ({l, r})}});
}

WeightExpr WeightAlgebra::subtract (const WeightExpr& l, const WeightExpr& r) {
  return isZero(r) ? WeightExpr(l) : WeightExpr::object ({{"-", WeightExpr::array ({l, r})}});
}

WeightExpr WeightAlgebra::power (const WeightExpr& a, const WeightExpr& b) {
  return isOne(b) ? WeightExpr(a) : (isZero(b) ? WeightExpr(true) : WeightExpr::object ({{"pow", WeightExpr::array ({a, b})}}));
}

WeightExpr WeightAlgebra::logOf (const WeightExpr& p) {
  return isOne(p) ? WeightExpr() : (opcode(p) == "exp" ? p.at("exp") : WeightExpr::object ({{"log", p}}));
}

WeightExpr WeightAlgebra::expOf (const WeightExpr& p) {
  return isZero(p) ? WeightExpr(true) : (opcode(p) == "log" ? p.at("log") : WeightExpr::object ({{"exp", p}}));
}

WeightExpr WeightAlgebra::multiply (const WeightExpr& l, const WeightExpr& r) {
  WeightExpr w;
  if (isOne(l))
    w = r;
  else if (isOne(r))
    w = l;
  else if (isZero(l) || isZero(r)) {
    // w = null
  } else if (l.is_number_integer() && r.is_number_integer())
    w = l.get<int>() * r.get<int>();
  else if (l.is_number() && r.is_number())
    w = l.get<double>() * r.get<double>();
  else
    w = WeightExpr::object ({{"*", WeightExpr::array({l,r})}});
  return w;
}

WeightExpr WeightAlgebra::add (const WeightExpr& l, const WeightExpr& r) {
  WeightExpr w;
  if (isZero(l))
    w = r;
  else if (isZero(r))
    w = l;
  else if (l.is_number_integer() && r.is_number_integer())
    w = l.get<int>() + r.get<int>();
  else if (l.is_number() && r.is_number())
    w = l.get<double>() + r.get<double>();
  else
    w = WeightExpr::object ({{"+", WeightExpr::array({l,r})}});
  return w;
}

bool WeightAlgebra::isZero (const WeightExpr& w) {
  return w.is_null()
    || (w.is_boolean() && !w.get<bool>())
    || (w.is_number_integer() && w.get<int>() == 0)
    || (w.is_number() && w.get<double>() == 0.);
}

bool WeightAlgebra::isOne (const WeightExpr& w) {
  return (w.is_boolean() && w.get<bool>())
    || (w.is_number_integer() && w.get<int>() == 1)
    || (w.is_number() && w.get<double>() == 1.);
}

string WeightAlgebra::opcode (const WeightExpr& w) {
  if (w.is_null())
    return string("null");
  if (w.is_boolean())
    return string("boolean");
  if (w.is_number_integer())
    return string("int");
  if (w.is_number())
    return string("float");
  if (w.is_string())
    return string("param");
  auto iter = w.begin();
  return iter.key();
}

const json& WeightAlgebra::operands (const WeightExpr& w) {
  auto iter = w.begin();
  return iter.value();
}

WeightExpr WeightAlgebra::bind (const WeightExpr& w, const ParamDefs& defs) {
  const string op = opcode(w);
  if (op == "null" || op == "boolean" || op == "int" || op == "float")
    return w;
  if (op == "param") {
    const string n = w.get<string>();
    if (!defs.count(n))
      return w;
    return defs.at(n);
  }
  if (op == "log" || op == "exp")
    return WeightExpr::object ({{op, bind (w.at(op), defs)}});
  json bindArgs;
  const json& args = operands(w);
  for (const auto& arg: args)
    bindArgs.push_back (bind (arg, defs));
  return WeightExpr::object ({{op, bindArgs}});
}

double WeightAlgebra::eval (const WeightExpr& w, const ParamDefs& defs) {
  const string op = opcode(w);
  if (op == "null") return 0;
  if (op == "boolean") return w.get<bool>() ? 1. : 0.;
  if (op == "int" || op == "float") return w.get<double>();
  if (op == "param") {
    const string n = w.get<string>();
    if (!defs.count(n))
      throw runtime_error(string("Parameter ") + n + (" not defined"));
    // optimize the special case that definition is a numeric assignment
    const auto& val = defs.at(n);
    if (val.is_number())
      return val.get<double>();
    return eval (defs.at(n), exclude(defs,n));
  }
  if (op == "log") return log (eval (w.at("log"), defs));
  if (op == "exp") return exp (eval (w.at("exp"), defs));
  vguard<double> evalArgs;
  const json& args = operands(w);
  for (const auto& arg: args)
    evalArgs.push_back (eval (arg, defs));
  if (op == "*") return evalArgs[0] * evalArgs[1];
  if (op == "/") return evalArgs[0] / evalArgs[1];
  if (op == "+") return evalArgs[0] + evalArgs[1];
  if (op == "-") return evalArgs[0] - evalArgs[1];
  if (op == "pow") return pow (evalArgs[0], evalArgs[1]);
  Abort("Unknown opcode: %s", op.c_str());
  return -numeric_limits<double>::infinity();
}

WeightExpr WeightAlgebra::deriv (const WeightExpr& w, const ParamDefs& defs, const string& param) {
  WeightExpr d;
  const string op = opcode(w);
  if (op == "null" || op == "boolean" || op == "int" || op == "float") {
    // d = null (implicit)
  } else if (op == "param") {
    const string n = w.get<string>();
    if (param == n)
      d = true;
    else if (defs.count(n))
      d = deriv (defs.at(n), exclude(defs,n), param);
    // else d = null (implicit)
  } else if (op == "exp") d = multiply (deriv (w.at("exp"), defs, param), w);  // w = exp(x), w' = x'exp(x)
  else if (op == "log") d = divide (deriv (w.at("log"), defs, param), w.at("log"));  // w = log(x), w' = x'/x
  else {
    const json& args = operands(w);
    vguard<WeightExpr> derivArgs;
    for (const auto& arg: args)
      derivArgs.push_back (deriv (arg, defs, param));
    if (op == "*") d = add (multiply(derivArgs[0],args[1]), multiply(args[0],derivArgs[1]));  // w = fg, w' = f'g + g'f
    else if (op == "/") d = subtract (divide(derivArgs[0],args[1]), multiply(derivArgs[1],divide(w,args[1])));  // w = f/g, w' = f'/g - g'f/g^2
    else if (op == "+") d = add (derivArgs[0], derivArgs[1]);  // w = f + g, w' = f' + g'
    else if (op == "-") d = subtract (derivArgs[0], derivArgs[1]);  // w = f - g, w' = f' - g'
    else if (op == "pow") d = multiply (w, add (multiply(derivArgs[1],logOf(args[0])), multiply(derivArgs[0],divide(args[1],args[0]))));  // w = a^b, w' = a^b (b'*log(a) + a'b/a)
    else
      Abort("Unknown opcode: %s", op.c_str());
  }
  return d;
}

set<string> WeightAlgebra::params (const WeightExpr& w, const ParamDefs& defs) {
  set<string> p;
  const string op = opcode(w);
  if (op == "null" || op == "boolean" || op == "int" || op == "float") {
    // p is empty
  } else if (op == "param") {
    const string n = w.get<string>();
    if (defs.count(n))
      p = params (defs.at(n), exclude(defs,n));
    else
      p.insert (n);
  } else if (op == "exp" || op == "log")
    p = params (w.at(op), defs);
  else {
    const json& args = operands(w);
    for (const auto& arg: args) {
      const set<string> argParams = params(arg,defs);
      p.insert (argParams.begin(), argParams.end());
    }
  }
  return p;
}

string WeightAlgebra::toString (const WeightExpr& w, const ParamDefs& defs, int parentPrecedence) {
  const string op = opcode(w);
  if (op == "null") return string("0");
  if (op == "boolean") return to_string (w.get<bool>() ? 1 : 0);
  if (op == "int") return to_string (w.get<int>());
  if (op == "float") return to_string (w.get<double>());
  if (op == "param") {
    const string n = w.get<string>();
    return defs.count(n) ? toString(defs.at(n),exclude(defs,n)) : n;
  }
  if (op == "log" || op == "exp") return op + "(" + toString(w.at(op),defs) + ")";
  const json& args = operands(w);
  if (op == "pow") return string("pow(") + toString(args[0],defs) + "," + toString(args[1],defs) + ")";

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
  if (op == "*") p = l = r = 2;
  else if (op == "/") { p = l = 2; r = 3; }
  else if (op == "-") { p = 1; l = 0; r = 2; }
  else if (op == "+") { p = 1; l = r = 0; }
  return string(parentPrecedence > p ? "(" : "")
    + toString(args[0],defs,l)
    + op
    + toString(args[1],defs,r)
    + (parentPrecedence > p ? ")" : "");
}

ParamDefs WeightAlgebra::exclude (const ParamDefs& defs, const string& param) {
  ParamDefs defsCopy (defs);
  defsCopy.erase (param);
  return defsCopy;
}

string WeightAlgebra::toJsonString (const ParamDefs& defs) {
  ostringstream out;
  out << "{";
  size_t n = 0;
  for (const auto& def: defs)
    out << (n++ ? "," : "") << "\"" << escaped_str(def.first) << "\":" << def.second;
  out << "}";
  return out.str();
}
