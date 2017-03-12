#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include "counts.h"
#include "backward.h"
#include "util.h"
#include "logger.h"

// Prefix for sqrt-transformed parameters
#define TransformedParamPrefix "$x"

// GSL multidimensional optimization parameters
#define StepSize 0.1
#define LineSearchTolerance 1e-4
#define EpsilonAbsolute 1e-3
#define MaxIterations 100

MachineCounts::MachineCounts (const EvaluatedMachine& machine) {
  init (machine);
}

MachineCounts::MachineCounts (const EvaluatedMachine& machine, const SeqPair& seqPair)
{
  init (machine);
  (void) add (machine, seqPair);
}

void MachineCounts::init (const EvaluatedMachine& machine) {
  count = vguard<vguard<double> > (machine.nStates());
  for (StateIndex s = 0; s < machine.nStates(); ++s)
    count[s].resize (machine.state[s].nTransitions);
}

double MachineCounts::add (const EvaluatedMachine& machine, const SeqPair& seqPair) {
  const ForwardMatrix forward (machine, seqPair);
  const BackwardMatrix backward (machine, seqPair);
  backward.getCounts (forward, *this);
  return forward.logLike();
}

MachineCounts& MachineCounts::operator+= (const MachineCounts& counts) {
  for (StateIndex s = 0; s < count.size(); ++s)
    for (size_t t = 0; t < count[s].size(); ++t)
      count[s][t] += counts.count[s][t];
  return *this;
}

void MachineCounts::writeJson (ostream& outs) const {
  vguard<string> s;
  for (const auto& c: count)
    s.push_back (string("[") + to_string_join (c, ",") + "]");
  outs << "[" << join (s, ",\n ") << "]" << endl;
}

MachineObjective::MachineObjective (const Machine& machine, const MachineCounts& counts, const Constraints& constraints) :
  constraints (constraints)
{
  for (StateIndex s = 0; s < machine.state.size(); ++s) {
    EvaluatedMachineState::TransIndex t = 0;
    for (TransList::const_iterator iter = machine.state[s].trans.begin();
	 iter != machine.state[s].trans.end(); ++iter, ++t)
      objective = WeightAlgebra::subtract (objective,
					   WeightAlgebra::multiply (counts.count[s][t],
								    WeightAlgebra::logOf ((*iter).weight)));
  }

  // p_i = (1 - exp(-x_i^2)) \prod_{k=1}^{i-1} exp(-x_k^2)
  const set<string> p = WeightAlgebra::params (objective, ParamDefs());
  int trIdx = 0;
  for (const auto& c: constraints.norm) {
    WeightExpr notPrev (true);
    for (size_t n = 0; n < c.size(); ++n) {
      const string& cParam = c[n];
      if (n == c.size() - 1)
	paramTransform[cParam] = notPrev;
      else {
	string trParam;
	do
	  trParam = string(TransformedParamPrefix) + to_string(++trIdx);
	while (p.count(trParam));

	transformedParamIndex[cParam] = param.size();
	param.push_back (trParam);
	WeightExpr notThis = WeightAlgebra::expOf (WeightAlgebra::multiply (-1,
									    WeightAlgebra::multiply (WeightExpr(trParam),
												     WeightExpr(trParam))));
	paramTransform[cParam] = WeightAlgebra::multiply (notPrev,
							  WeightAlgebra::subtract (WeightExpr(true), notThis));
	notPrev = WeightAlgebra::multiply (notPrev, notThis);
      }
    }
  }

  deriv.reserve (param.size());
  for (const auto& p: param)
    deriv.push_back (WeightAlgebra::deriv (objective, paramTransform, p));

  LogThisAt (5, toString());
}

string MachineObjective::toString() const {
  string s = string("E = ") + WeightAlgebra::toString(objective,paramTransform) + "\n";
  for (size_t n = 0; n < param.size(); ++n)
    s += "dE/d" + param[n] + " = " + WeightAlgebra::toString(deriv[n],paramTransform) + "\n";
  return s;
}

Params gsl_vector_to_params (const gsl_vector *v, const MachineObjective& ml) {
  Params p;
  p.defs = ml.paramTransform;
  for (size_t n = 0; n < ml.param.size(); ++n)
    p.defs[ml.param[n]] = WeightExpr (gsl_vector_get (v, n));
  return p;
}

double gsl_machine_objective (const gsl_vector *v, void *voidML)
{
  const MachineObjective& ml (*((MachineObjective*)voidML));
  const Params pv = gsl_vector_to_params (v, ml);

  const double f = WeightAlgebra::eval (ml.objective, pv.defs);

  LogThisAt (4, pv.toJsonString() << endl);
  LogThisAt (5, "gsl_machine_objective(" << to_string_join(gsl_vector_to_stl(v)) << ") = " << f << endl);

  return f;
}

void gsl_machine_objective_deriv (const gsl_vector *v, void *voidML, gsl_vector *df)
{
  const MachineObjective& ml (*((MachineObjective*)voidML));
  const Params pv = gsl_vector_to_params (v, ml);

  for (size_t n = 0; n < ml.param.size(); ++n)
    gsl_vector_set (df, n, WeightAlgebra::eval (ml.deriv[n], pv.defs));

  const vguard<double> v_stl = gsl_vector_to_stl(v), df_stl = gsl_vector_to_stl(df);
  LogThisAt (5, "gsl_machine_objective_deriv(" << to_string_join(v_stl) << ") = (" << to_string_join(df_stl) << ")" << endl);
}

void gsl_machine_objective_with_deriv (const gsl_vector *x, void *voidML, double *f, gsl_vector *df)
{
  *f = gsl_machine_objective (x, voidML);
  gsl_machine_objective_deriv (x, voidML, df);
}

Params MachineObjective::optimize (const Params& seed) const {
  gsl_vector *v;
  gsl_multimin_function_fdf func;
  func.n = param.size();
  func.f = gsl_machine_objective;
  func.df = gsl_machine_objective_deriv;
  func.fdf = gsl_machine_objective_with_deriv;
  func.params = (void*) this;

  gsl_vector* x = gsl_vector_alloc (func.n);
  // p_i = (1 - z_i) \prod_{k=1}^{i-1} z_k
  // where z_i = exp(-x_i^2)
  // \prod_{k=1}^{i-1} z_k = 1 - \sum_{k=1}^{i-1} p_k
  // so z_i = 1 - p_i / (1 - \sum_{k=1}^{i-1} p_k)
  // x_i = sqrt(-log z_i)
  for (const auto& c: constraints.norm) {
    double pSum = 0;
    for (size_t n = 0; n + 1 < c.size(); ++n) {
      const string& cParam = c[n];
      const double p = seed.defs.at(cParam).get<double>();
      const double z = 1 - p / (1 - pSum);
      pSum += p;
      gsl_vector_set (x, transformedParamIndex.at(cParam), sqrt (-log (z)));
    }
  }

  const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_vector_bfgs2;
  gsl_multimin_fdfminimizer *s = gsl_multimin_fdfminimizer_alloc (T, func.n);

  gsl_multimin_fdfminimizer_set (s, &func, x, StepSize, LineSearchTolerance);
  
  size_t iter = 0;
  int status;
  do
    {
      iter++;
      status = gsl_multimin_fdfminimizer_iterate (s);

      const vguard<double> x_stl = gsl_vector_to_stl(s->x);
      LogThisAt (5, "iteration #" << iter << ": x=(" << to_string_join(x_stl) << ")" << endl);

      if (status)
        break;

      status = gsl_multimin_test_gradient (s->gradient, EpsilonAbsolute);
    }
  while (status == GSL_CONTINUE && iter < MaxIterations);

  const Params finalTransformedParams = gsl_vector_to_params (s->x, *this);
  Params finalParams = seed;
  for (const auto& paramFunc: paramTransform)
    finalParams.defs[paramFunc.first] = WeightAlgebra::eval (paramFunc.second, finalTransformedParams.defs);
  
  gsl_multimin_fdfminimizer_free (s);
  gsl_vector_free (x);

  return finalParams;
}
