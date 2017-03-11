#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include "counts.h"
#include "backward.h"
#include "util.h"

#define LagrangeMultiplierPrefix "LagrangeMultiplier"

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

  const auto p = WeightAlgebra::params (lagrangian);
  param.insert (param.end(), p.begin(), p.end());

  int lm = 0;
  for (const auto& c: constraints.norm) {
    string lambda;
    do
      lambda = string(LagrangeMultiplierPrefix) + to_string(lm++);
    while (p.count(lambda));
    lagrangeMultiplier.push_back (lambda);

    TransWeight cSum;
    for (const auto& cParam: c)
      cSum = WeightAlgebra::add (cSum, cParam);
    lagrangian = WeightAlgebra::add (lagrangian,
				     WeightAlgebra::multiply (lambda,
							      WeightAlgebra::subtract (true, cSum)));
  }

  paramDeriv.reserve (param.size());
  multiplierDeriv.reserve (lagrangeMultiplier.size());
  for (const auto& p: param)
    paramDeriv.push_back (WeightAlgebra::deriv (lagrangian, p));
  for (const auto& lambda: lagrangeMultiplier)
    multiplierDeriv.push_back (WeightAlgebra::deriv (lagrangian, lambda));
}

Params gsl_vector_to_params (const gsl_vector *v, const MachineLagrangian& ml) {
  Params p;

  // acidbot_param_n = exp (gsl_param_n)
  for (size_t n = 0; n < ml.param.size(); ++n)
    p.param[ml.param[n]] = exp (gsl_vector_get (v, n));

  for (size_t n = 0; n < ml.lagrangeMultiplier.size(); ++n)
    p.param[ml.param[n]] = gsl_vector_get (v, n + ml.param.size());

  return p;
}

double gsl_machine_lagrangian (const gsl_vector *v, void *voidML)
{
  const MachineLagrangian& ml (*((MachineLagrangian*)voidML));
  const Params pv = gsl_vector_to_params (v, ml);
  return WeightAlgebra::eval (ml.lagrangian, pv);
}

void gsl_machine_lagrangian_deriv (const gsl_vector *v, void *voidML, gsl_vector *df)
{
  const MachineLagrangian& ml (*((MachineLagrangian*)voidML));
  const Params pv = gsl_vector_to_params (v, ml);

  // acidbot_param_n = exp (gsl_param_n)
  // so d(lagrangian)/d(gsl_param_n) = d(lagrangian)/d(acidbot_param_n) * acidbot_param_n
  for (size_t n = 0; n < ml.param.size(); ++n)
    gsl_vector_set (df, n, WeightAlgebra::eval (ml.paramDeriv[n], pv) * pv.param.at (ml.param[n]));

  for (size_t n = 0; n < ml.lagrangeMultiplier.size(); ++n)
    gsl_vector_set (df, n + ml.param.size(), WeightAlgebra::eval (ml.multiplierDeriv[n], pv));
}

void gsl_machine_lagrangian_with_deriv (const gsl_vector *x, void *voidML, double *f, gsl_vector *df)
{
  *f = gsl_machine_lagrangian (x, voidML);
  gsl_machine_lagrangian_deriv (x, voidML, df);
}

#define StepSize 0.01
#define LineSearchTolerance 1e-4
#define EpsilonAbsolute 1e-3
#define MaxIterations 100
Params MachineLagrangian::optimize (const Params& seed) const {
  gsl_vector *v;
  gsl_multimin_function_fdf func;
  func.n = param.size() + lagrangeMultiplier.size();
  func.f = gsl_machine_lagrangian;
  func.df = gsl_machine_lagrangian_deriv;
  func.fdf = gsl_machine_lagrangian_with_deriv;
  func.params = (void*) this;

  gsl_vector* x = gsl_vector_alloc (func.n);
  // acidbot_param_n = exp (gsl_param_n)
  // so gsl_param_n = log (acidbot_param_n)
  for (size_t n = 0; n < param.size(); ++n)
    gsl_vector_set (x, n, log (seed.param.at (param[n])));

  for (size_t n = 0; n < lagrangeMultiplier.size(); ++n)
    gsl_vector_set (x, n + param.size(), 0.);

  const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_vector_bfgs2;
  gsl_multimin_fdfminimizer *s = gsl_multimin_fdfminimizer_alloc (T, func.n);

  gsl_multimin_fdfminimizer_set (s, &func, x, StepSize, LineSearchTolerance);
  
  size_t iter = 0;
  int status;
  do
    {
      iter++;
      status = gsl_multimin_fdfminimizer_iterate (s);

      if (status)
        break;

      status = gsl_multimin_test_gradient (s->gradient, EpsilonAbsolute);
    }
  while (status == GSL_CONTINUE && iter < MaxIterations);

  const Params finalParams = gsl_vector_to_params (x, *this);

  gsl_multimin_fdfminimizer_free (s);
  gsl_vector_free (x);

  return finalParams;
}
