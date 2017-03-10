#include <math.h>
#include "weight.h"
#include "logsumexp.h"
#include "util.h"

TransWeight WeightAlgebra::geometricSum (const TransWeight& p) {
  return TransWeight::object ({{"/", TransWeight::array ({1, TransWeight::object ({{"-", TransWeight::array ({1,p})}})})}});
}

TransWeight WeightAlgebra::divide (const TransWeight& l, const TransWeight& r) {
  return TransWeight::object ({{"/", TransWeight::array ({l, r})}});
}

TransWeight WeightAlgebra::subtract (const TransWeight& l, const TransWeight& r) {
  return TransWeight::object ({{"-", TransWeight::array ({l, r})}});
}

TransWeight WeightAlgebra::pow (const TransWeight& a, const TransWeight& b) {
  return TransWeight::object ({{"^", TransWeight::array ({a, b})}});
}

TransWeight WeightAlgebra::logOf (const TransWeight& p) {
  return TransWeight::object ({{"log", p}});
}

TransWeight WeightAlgebra::expOf (const TransWeight& p) {
  return TransWeight::object ({{"exp", p}});
}

TransWeight WeightAlgebra::multiply (const TransWeight& l, const TransWeight& r) {
  TransWeight w;
  if (l.is_boolean() && l.get<bool>())
    w = r;
  else if (r.is_boolean() && r.get<bool>())
    w = l;
  else if (l.is_number_integer() && r.is_number_integer())
    w = l.get<int>() * r.get<int>();
  else if (l.is_number() && r.is_number())
    w = l.get<double>() * r.get<double>();
  else
    w = TransWeight::object ({{"*", TransWeight::array({l,r})}});
  return w;
}

TransWeight WeightAlgebra::add (const TransWeight& l, const TransWeight& r) {
  TransWeight w;
  if (l.is_null())
    w = r;
  else if (r.is_null())
    w = l;
  else if (l.is_number_integer() && r.is_number_integer())
    w = l.get<int>() + r.get<int>();
  else if (l.is_number() && r.is_number())
    w = l.get<double>() + r.get<double>();
  else
    w = TransWeight::object ({{"+", TransWeight::array({l,r})}});
  return w;
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
  vguard<double> evalArgs;
  const json& args = operands(w);
  for (const auto& arg: args)
    evalArgs.push_back (eval (arg, params));
  if (op == "*") return evalArgs[0] * evalArgs[1];
  if (op == "/") return evalArgs[0] / evalArgs[1];
  if (op == "+") return evalArgs[0] + evalArgs[1];
  if (op == "-") return evalArgs[0] - evalArgs[1];
  if (op == "^") return pow (evalArgs[0], evalArgs[1]);
  if (op == "log") return log (evalArgs[0]);
  if (op == "exp") return exp (evalArgs[0]);
  Abort("Unknown opcode: %s", op.c_str());
  return -numeric_limits<double>::infinity();
}

TransWeight WeightAlgebra::deriv (const TransWeight& w, const string& param) {
  TransWeight d;
  const string op = opcode(w);
  if (op == "null" || op == "boolean" || op == "int" || op == "float")
    d = 0;
  else if (op == "param")
    d = (param == w.get<string>() ? 1. : 0.);
  else {
    const json& args = operands(w);
    vguard<TransWeight> derivArgs;
    for (const auto& arg: args)
      derivArgs.push_back (deriv (arg, param));
    if (op == "*") d = add (multiply(derivArgs[0],args[1]), multiply(args[0],derivArgs[1]));  // w = fg, w' = f'g + g'f
    else if (op == "/") d = subtract (divide(derivArgs[0],args[1]), multiply(derivArgs[1],divide(w,args[0])));  // w = f/g, w' = f'/g - g'f/g^2
    else if (op == "+") d = add (derivArgs[0], derivArgs[1]);  // w = f + g, w' = f' + g'
    else if (op == "-") d = subtract (derivArgs[0], derivArgs[1]);  // w = f - g, w' = f' - g'
    else if (op == "exp") d = multiply (derivArgs[0], w);  // w = exp(x), w' = x'exp(x)
    else if (op == "log") d = divide (derivArgs[0], args[0]);  // w = log(x), w' = x'/x
    else if (op == "^") d = multiply (w, add (multiply(derivArgs[1],logOf(args[0])), multiply(derivArgs[0],divide(args[1],args[0]))));  // w = a^b, w' = a^b (b'*log(a) + a'b/a)
    else
      Abort("Unknown opcode: %s", op.c_str());
  }
  return d;
}
