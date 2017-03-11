#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include "counts.h"
#include "backward.h"
#include "util.h"

// Prefix for Lagrange multiplier parameters
#define LagrangeMultiplierPrefix "$lagrange"

// Prefix for sqrt-transformed parameters
#define TransformedParamPrefix "$param"

// GSL multidimensional optimization parameters
#define StepSize 0.1
#define LineSearchTolerance 1e-4
#define EpsilonAbsolute 1e-3
#define MaxIterations 100

MachineCounts::MachineCounts (const EvaluatedMachine& machine, const SeqPair& seqPair) :
  count (machine.nStates())
{
  for (StateIndex s = 0; s < machine.nStates(); ++s)
    count[s].resize (machine.state[s].nTransitions);

  const ForwardMatrix forward (machine, seqPair);
  const BackwardMatrix backward (machine, seqPair);

  backward.getCounts (forward, *this);
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

MachineLagrangian::MachineLagrangian (const Machine& machine, const MachineCounts& counts, const Constraints& constraints) {

  for (StateIndex s = 0; s < machine.state.size(); ++s) {
    EvaluatedMachineState::TransIndex t = 0;
    for (TransList::const_iterator iter = machine.state[s].trans.begin();
	 iter != machine.state[s].trans.end(); ++iter, ++t)
      lagrangian = WeightAlgebra::add (lagrangian,
				       WeightAlgebra::multiply (counts.count[s][t],
								WeightAlgebra::logOf ((*iter).weight)));
  }

  const set<string> p = WeightAlgebra::params (lagrangian, ParamDefs());
  int lmIdx = 0, trIdx = 0;
  for (const auto& c: constraints.norm) {
    string lmParam;
    do
      lmParam = string(LagrangeMultiplierPrefix) + to_string(++lmIdx);
    while (p.count(lmParam));
    param.push_back (lmParam);

    WeightExpr cSum;
    for (const auto& cParam: c) {
      cSum = WeightAlgebra::add (cSum, cParam);

      string trParam;
      do
	trParam = string(TransformedParamPrefix) + to_string(++trIdx);
      while (p.count(trParam));

      transformedParamIndex[cParam] = param.size();
      param.push_back (trParam);
      paramTransform[cParam] = WeightAlgebra::multiply (WeightExpr(trParam), WeightExpr(trParam));
    }

    lagrangian = WeightAlgebra::add (lagrangian,
				     WeightAlgebra::multiply (lmParam,
							      WeightAlgebra::subtract (true, cSum)));
  }

  for (const auto& p: param) {
    const WeightExpr d = WeightAlgebra::deriv (lagrangian, paramTransform, p);
    gradSquared = WeightAlgebra::add (gradSquared, WeightAlgebra::multiply (d, d));
  }

  deriv.reserve (param.size());
  for (const auto& p: param)
    deriv.push_back (WeightAlgebra::deriv (gradSquared, paramTransform, p));

  cerr << "L = " << WeightAlgebra::toString(lagrangian,paramTransform) << endl;
  cerr << "(grad L)^2 = " << WeightAlgebra::toString(gradSquared,paramTransform) << endl;
  for (size_t n = 0; n < param.size(); ++n)
    cerr << "d(grad L)^2/d" << param[n] << " = " << WeightAlgebra::toString(deriv[n],paramTransform) << endl;
}

Params gsl_vector_to_params (const gsl_vector *v, const MachineLagrangian& ml) {
  Params p;
  p.defs = ml.paramTransform;
  for (size_t n = 0; n < ml.param.size(); ++n)
    p.defs[ml.param[n]] = WeightExpr (gsl_vector_get (v, n));
  return p;
}

double gsl_grad2_machine_lagrangian (const gsl_vector *v, void *voidML)
{
  const MachineLagrangian& ml (*((MachineLagrangian*)voidML));
  const Params pv = gsl_vector_to_params (v, ml);

  const double g2 = WeightAlgebra::eval (ml.gradSquared, pv.defs);

  pv.writeJson(cerr);
  const vguard<double> v_stl = gsl_vector_to_stl(v);
  cerr << "gsl_grad2_machine_lagrangian(" << to_string_join(v_stl) << ") = " << g2 << endl;

  return g2;
}

void gsl_grad2_machine_lagrangian_deriv (const gsl_vector *v, void *voidML, gsl_vector *df)
{
  const MachineLagrangian& ml (*((MachineLagrangian*)voidML));
  const Params pv = gsl_vector_to_params (v, ml);

  for (size_t n = 0; n < ml.param.size(); ++n)
    gsl_vector_set (df, n, WeightAlgebra::eval (ml.deriv[n], pv.defs));

  const vguard<double> v_stl = gsl_vector_to_stl(v), df_stl = gsl_vector_to_stl(df);
  cerr << "gsl_grad2_machine_lagrangian_deriv(" << to_string_join(v_stl) << ") = (" << to_string_join(df_stl) << ")" << endl;
}

void gsl_grad2_machine_lagrangian_with_deriv (const gsl_vector *x, void *voidML, double *f, gsl_vector *df)
{
  *f = gsl_grad2_machine_lagrangian (x, voidML);
  gsl_grad2_machine_lagrangian_deriv (x, voidML, df);
}

Params MachineLagrangian::optimize (const Params& seed) const {
  gsl_vector *v;
  gsl_multimin_function_fdf func;
  func.n = param.size();
  func.f = gsl_grad2_machine_lagrangian;
  func.df = gsl_grad2_machine_lagrangian_deriv;
  func.fdf = gsl_grad2_machine_lagrangian_with_deriv;
  func.params = (void*) this;

  gsl_vector* x = gsl_vector_alloc (func.n);
  for (size_t n = 0; n < func.n; ++n)
    gsl_vector_set (x, n, 1);
  // acidbot_param_n = (gsl_param_n)^2
  // so gsl_param_n = sqrt(acidbot_param_n)
  for (const auto& paramIdx: transformedParamIndex)
    gsl_vector_set (x, paramIdx.second, sqrt (seed.defs.at(paramIdx.first).get<double>()));

  const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_vector_bfgs2;
  gsl_multimin_fdfminimizer *s = gsl_multimin_fdfminimizer_alloc (T, func.n);

  gsl_multimin_fdfminimizer_set (s, &func, x, StepSize, LineSearchTolerance);
  
  size_t iter = 0;
  int status;
  do
    {
      cerr << endl;
      iter++;
      status = gsl_multimin_fdfminimizer_iterate (s);

      const vguard<double> x_stl = gsl_vector_to_stl(x);
      cerr << "iteration #" << iter << ": x=(" << to_string_join(x_stl) << ")" << endl;

      if (status)
        break;

      status = gsl_multimin_test_gradient (s->gradient, EpsilonAbsolute);
    }
  while (status == GSL_CONTINUE && iter < MaxIterations);

  const Params finalTransformedParams = gsl_vector_to_params (x, *this);
  Params finalParams = seed;
  for (const auto& paramFunc: paramTransform)
    finalParams.defs[paramFunc.first] = WeightAlgebra::eval (paramFunc.second, finalTransformedParams.defs);
  
  gsl_multimin_fdfminimizer_free (s);
  gsl_vector_free (x);

  return finalParams;
}
