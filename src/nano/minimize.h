#ifndef MINIMIZE_INCLUDED
#define MINIMIZE_INCLUDED

#include <gsl/gsl_vector.h>
#include "../vguard.h"
#include "../weight.h"

struct Minimizer {
  double stepSize, lineSearchTolerance, epsilonAbsolute;
  int maxIterations;

  WeightExpr func;
  vguard<string> paramName;
  vguard<WeightExpr> deriv;

  Minimizer (const WeightExpr& f);
  ParamDefs minimize (const ParamDefs& seed) const;
  
  ParamDefs gsl_vector_to_params (const gsl_vector *v) const;

  static double gsl_objective (const gsl_vector *v, void* minimizer);
  static void gsl_objective_deriv (const gsl_vector *v, void* minimizer, gsl_vector *df);
  static void gsl_objective_with_deriv (const gsl_vector *x, void* minimizer, double *f, gsl_vector *df);
};

#endif /* MINIMIZE_INCLUDED */

