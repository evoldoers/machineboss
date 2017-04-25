#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include "minimize.h"
#include "../logsumexp.h"
#include "../logger.h"

// GSL multidimensional minimization parameters
#define DefaultStepSize 0.1
#define DefaultLineSearchTolerance 1e-4
#define DefaultEpsilonAbsolute 1e-3
#define DefaultMaxIterations 100

Minimizer::Minimizer (const WeightExpr& f) :
  stepSize (DefaultStepSize),
  lineSearchTolerance (DefaultLineSearchTolerance),
  epsilonAbsolute (DefaultEpsilonAbsolute),
  maxIterations (DefaultMaxIterations),
  func (f)
{
  const auto p = WeightAlgebra::params (f, ParamDefs());
  paramName = vguard<string> (p.begin(), p.end());
  for (const auto& n: paramName)
    deriv.push_back (WeightAlgebra::deriv (func, ParamDefs(), n));
}

ParamDefs Minimizer::gsl_vector_to_params (const gsl_vector *v) const {
  ParamDefs p;
  for (size_t n = 0; n < paramName.size(); ++n)
    p[paramName[n]] = WeightExpr (gsl_vector_get (v, n));
  return p;
}

double Minimizer::gsl_objective (const gsl_vector *v, void* voidMin) {
  const Minimizer& minimizer (*(Minimizer*)voidMin);
  const ParamDefs p = minimizer.gsl_vector_to_params(v);

  const double f = WeightAlgebra::eval (minimizer.func, p);

  LogThisAt (7, WeightAlgebra::toJsonString(p) << endl);
  LogThisAt (7, "gsl_objective(" << to_string_join(gsl_vector_to_stl(v)) << ") = " << f << endl);

  return f;
}

void Minimizer::gsl_objective_deriv (const gsl_vector *v, void* voidMin, gsl_vector *df) {
  const Minimizer& minimizer (*(Minimizer*)voidMin);
  const ParamDefs p = minimizer.gsl_vector_to_params(v);

  for (size_t n = 0; n < minimizer.paramName.size(); ++n)
    gsl_vector_set (df, n, WeightAlgebra::eval (minimizer.deriv[n], p));

  const vguard<double> v_stl = gsl_vector_to_stl(v), df_stl = gsl_vector_to_stl(df);
  LogThisAt (7, "gsl_objective_deriv(" << to_string_join(v_stl) << ") = (" << to_string_join(df_stl) << ")" << endl);
}

void Minimizer::gsl_objective_with_deriv (const gsl_vector *x, void* voidMin, double *f, gsl_vector *df) {
  *f = gsl_objective (x, voidMin);
  gsl_objective_deriv (x, voidMin, df);
}
  
ParamDefs Minimizer::minimize (const ParamDefs& seed) const {
  gsl_vector *v;
  gsl_multimin_function_fdf func;
  func.n = paramName.size();
  func.f = gsl_objective;
  func.df = gsl_objective_deriv;
  func.fdf = gsl_objective_with_deriv;
  func.params = (void*) this;

  gsl_vector* x = gsl_vector_alloc (func.n);
  for (size_t n = 0; n < paramName.size(); ++n)
    gsl_vector_set (x, n, seed.at(paramName[n]).get<double>());

  const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_vector_bfgs2;
  gsl_multimin_fdfminimizer *s = gsl_multimin_fdfminimizer_alloc (T, func.n);

  gsl_multimin_fdfminimizer_set (s, &func, x, stepSize, lineSearchTolerance);
  
  size_t iter = 0;
  int status;
  do
    {
      iter++;
      status = gsl_multimin_fdfminimizer_iterate (s);

      const vguard<double> x_stl = gsl_vector_to_stl(s->x);
      LogThisAt (6, "iteration #" << iter << ": x=(" << to_string_join(x_stl) << ")" << endl);

      if (status)
        break;

      status = gsl_multimin_test_gradient (s->gradient, epsilonAbsolute);
    }
  while (status == GSL_CONTINUE && iter < maxIterations);

  const ParamDefs finalParams = gsl_vector_to_params (s->x);
  
  gsl_multimin_fdfminimizer_free (s);
  gsl_vector_free (x);

  return finalParams;
}
