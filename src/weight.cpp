#include <math.h>
#include "weight.h"
#include "logsumexp.h"
#include "util.h"

TransWeight WeightAlgebra::geometricSum (const TransWeight& p) {
  return TransWeight::object ({{"/", TransWeight::array ({true, TransWeight::object ({{"-", TransWeight::array ({true, p})}})})}});
}

TransWeight WeightAlgebra::divide (const TransWeight& l, const TransWeight& r) {
  return isOne(r) ? TransWeight(l) : TransWeight::object ({{"/", TransWeight::array ({l, r})}});
}

TransWeight WeightAlgebra::subtract (const TransWeight& l, const TransWeight& r) {
  return isZero(r) ? TransWeight(l) : TransWeight::object ({{"-", TransWeight::array ({l, r})}});
}

TransWeight WeightAlgebra::power (const TransWeight& a, const TransWeight& b) {
  return isOne(b) ? TransWeight(a) : (isZero(b) ? TransWeight(true) : TransWeight::object ({{"pow", TransWeight::array ({a, b})}}));
}

TransWeight WeightAlgebra::logOf (const TransWeight& p) {
  return isOne(p) ? TransWeight() : TransWeight::object ({{"log", p}});
}

TransWeight WeightAlgebra::expOf (const TransWeight& p) {
  return isZero(p) ? TransWeight(true) : TransWeight::object ({{"exp", p}});
}

TransWeight WeightAlgebra::multiply (const TransWeight& l, const TransWeight& r) {
  TransWeight w;
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
    w = TransWeight::object ({{"*", TransWeight::array({l,r})}});
  return w;
}

TransWeight WeightAlgebra::add (const TransWeight& l, const TransWeight& r) {
  TransWeight w;
  if (isZero(l))
    w = r;
  else if (isZero(r))
    w = l;
  else if (l.is_number_integer() && r.is_number_integer())
    w = l.get<int>() + r.get<int>();
  else if (l.is_number() && r.is_number())
    w = l.get<double>() + r.get<double>();
  else
    w = TransWeight::object ({{"+", TransWeight::array({l,r})}});
  return w;
}

bool WeightAlgebra::isZero (const TransWeight& w) {
  return w.is_null()
    || (w.is_boolean() && !w.get<bool>())
    || (w.is_number_integer() && w.get<int>() == 0)
    || (w.is_number() && w.get<double>() == 0.);
}

bool WeightAlgebra::isOne (const TransWeight& w) {
  return (w.is_boolean() && w.get<bool>())
    || (w.is_number_integer() && w.get<int>() == 1)
    || (w.is_number() && w.get<double>() == 1.);
}

string WeightAlgebra::opcode (const TransWeight& w) {
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

const json& WeightAlgebra::operands (const TransWeight& w) {
  auto iter = w.begin();
  return iter.value();
}

double WeightAlgebra::eval (const TransWeight& w, const Params& params) {
  const string op = opcode(w);
  if (op == "null") return 0;
  if (op == "boolean") return w.get<bool>() ? 1. : 0.;
  if (op == "int" || op == "float") return w.get<double>();
  if (op == "param") return params.param.at (w.get<string>());
  if (op == "log") return log (eval (w.at("log"), params));
  if (op == "exp") return exp (eval (w.at("exp"), params));
  vguard<double> evalArgs;
  const json& args = operands(w);
  for (const auto& arg: args)
    evalArgs.push_back (eval (arg, params));
  if (op == "*") return evalArgs[0] * evalArgs[1];
  if (op == "/") return evalArgs[0] / evalArgs[1];
  if (op == "+") return evalArgs[0] + evalArgs[1];
  if (op == "-") return evalArgs[0] - evalArgs[1];
  if (op == "pow") return pow (evalArgs[0], evalArgs[1]);
  Abort("Unknown opcode: %s", op.c_str());
  return -numeric_limits<double>::infinity();
}

TransWeight WeightAlgebra::deriv (const TransWeight& w, const string& param) {
  TransWeight d;
  const string op = opcode(w);
  if (op == "null" || op == "boolean" || op == "int" || op == "float") {
    // d = null
  } else if (op == "param") {
    if (param == w.get<string>())
      d = true;
    // else d = null
  } else if (op == "exp") d = multiply (deriv (w.at("exp"), param), w);  // w = exp(x), w' = x'exp(x)
  else if (op == "log") d = divide (deriv (w.at("log"), param), w.at("log"));  // w = log(x), w' = x'/x
  else {
    const json& args = operands(w);
    vguard<TransWeight> derivArgs;
    for (const auto& arg: args)
      derivArgs.push_back (deriv (arg, param));
    if (op == "*") d = add (multiply(derivArgs[0],args[1]), multiply(args[0],derivArgs[1]));  // w = fg, w' = f'g + g'f
    else if (op == "/") d = subtract (divide(derivArgs[0],args[1]), multiply(derivArgs[1],divide(w,args[0])));  // w = f/g, w' = f'/g - g'f/g^2
    else if (op == "+") d = add (derivArgs[0], derivArgs[1]);  // w = f + g, w' = f' + g'
    else if (op == "-") d = subtract (derivArgs[0], derivArgs[1]);  // w = f - g, w' = f' - g'
    else if (op == "pow") d = multiply (w, add (multiply(derivArgs[1],logOf(args[0])), multiply(derivArgs[0],divide(args[1],args[0]))));  // w = a^b, w' = a^b (b'*log(a) + a'b/a)
    else
      Abort("Unknown opcode: %s", op.c_str());
  }
  return d;
}

set<string> WeightAlgebra::params (const TransWeight& w) {
  set<string> p;
  const string op = opcode(w);
  if (op == "null" || op == "boolean" || op == "int" || op == "float") {
    // p is empty
  } else if (op == "param")
    p.insert (w.get<string>());
  else if (op == "exp" || op == "log")
    p = params (w.at(op));
  else {
    const json& args = operands(w);
    for (const auto& arg: args) {
      const set<string> argParams = params(arg);
      p.insert (argParams.begin(), argParams.end());
    }
  }
  return p;
}
