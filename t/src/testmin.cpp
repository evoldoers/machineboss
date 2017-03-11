#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include "../../src/logsumexp.h"
#include "../../src/util.h"

// GSL multidimensional optimization parameters
#define StepSize 0.01
#define LineSearchTolerance 1e-4
#define EpsilonAbsolute 1e-3
#define MaxIterations 100

double gsl_test_lagrangian (const gsl_vector *v, void *voidML)
{
  const double a = gsl_vector_get(v,0), b = gsl_vector_get(v,1), lambda = gsl_vector_get(v,2);
  // const double p = exp(-a*a), q = exp(-b*b);
  // const double l = -2*log(p) - log(q) - lambda*(1-p-q)
  const double l = 2*a*a + b*b - lambda*(1-exp(-a*a)-exp(-b*b));

  const vguard<double> v_stl = gsl_vector_to_stl(v);
  cerr << "f(" << to_string_join(v_stl) << ") = " << l << endl;

  return l;
}

void gsl_test_lagrangian_deriv (const gsl_vector *v, void *voidML, gsl_vector *df)
{
  const double a = gsl_vector_get(v,0), b = gsl_vector_get(v,1), lambda = gsl_vector_get(v,2);

  gsl_vector_set (df, 0, 4*a - lambda*2*a*exp(-a*a));
  gsl_vector_set (df, 1, 2*b - lambda*2*b*exp(-b*b));
  gsl_vector_set (df, 2, -(1-exp(-a*a)-exp(-b*b)));

  const vguard<double> v_stl = gsl_vector_to_stl(v), df_stl = gsl_vector_to_stl(df);
  cerr << "df(" << to_string_join(v_stl) << ") = (" << to_string_join(df_stl) << ")" << endl;
}

void gsl_test_lagrangian_with_deriv (const gsl_vector *x, void *voidML, double *f, gsl_vector *df)
{
  *f = gsl_test_lagrangian (x, voidML);
  gsl_test_lagrangian_deriv (x, voidML, df);
}

int main (int argc, char** argv) {
  if (argc != 4) {
    cerr << "Usage: " << argv[0] << " p q lambda" << endl;
    exit(1);
  }

  gsl_vector *v;
  gsl_multimin_function_fdf func;
  func.n = 3;
  func.f = gsl_test_lagrangian;
  func.df = gsl_test_lagrangian_deriv;
  func.fdf = gsl_test_lagrangian_with_deriv;
  func.params = NULL;

  gsl_vector* x = gsl_vector_alloc (func.n);
  gsl_vector_set (x, 0, sqrt (-log (atof (argv[1]))));
  gsl_vector_set (x, 1, sqrt (-log (atof (argv[2]))));
  gsl_vector_set (x, 2, atof (argv[3]));

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

  gsl_multimin_fdfminimizer_free (s);
  gsl_vector_free (x);

  return 0;
}
