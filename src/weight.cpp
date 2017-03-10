#include "weight.h"
#include "logsumexp.h"
#include "util.h"

TransWeight WeightAlgebra::geometricSum (const TransWeight& p) {
  return TransWeight::object ({{"/", TransWeight::array ({1, TransWeight::object ({{"-", TransWeight::array ({1,p})}})})}});
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


double WeightAlgebra::evalLog (const TransWeight& w, const Params& params) {
  const string op = opcode(w);
  if (op == "null") return -numeric_limits<double>::infinity();
  if (op == "boolean") return w.get<bool>() ? 0. : -numeric_limits<double>::infinity();
  if (op == "int" || op == "float") return log (w.get<double>());
  if (op == "param") return log (params.param.at (w.get<string>()));
  vguard<double> evalLogArgs;
  const json& args = operands(w);
  for (const auto& arg: args)
    evalLogArgs.push_back (evalLog (arg, params));
  if (op == "*") return evalLogArgs[0] + evalLogArgs[1];
  if (op == "/") return evalLogArgs[0] - evalLogArgs[1];
  if (op == "+") return log_sum_exp (evalLogArgs[0], evalLogArgs[1]);
  if (op == "-") return log_subtract_exp (evalLogArgs[0], evalLogArgs[1]);
  Abort("Unknown opcode: %s", op.c_str());
  return -numeric_limits<double>::infinity();
}

TransWeight WeightAlgebra::logDerivLog (const TransWeight& w, const string& param) {
  // WRITE ME
  return TransWeight();
}
