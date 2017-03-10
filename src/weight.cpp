#include "weight.h"


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
  else if (w.is_number())
    return string("number");
  else if (w.is_boolean())
    return string("boolean");
  auto iter = w.begin();
  return iter.key();
}

const json& WeightAlgebra::operands (const TransWeight& w) {
  auto iter = w.begin();
  return iter.value();
}
