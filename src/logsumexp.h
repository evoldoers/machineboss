#ifndef LOGSUMEXP_INCLUDED
#define LOGSUMEXP_INCLUDED

#include <limits>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <cmath>
#include "vguard.h"

using namespace std;

/* uncomment to disable lookup table */
/*
#define LOG_SUM_EXP_SLOW
*/

/* uncomment to catch NaN errors */
/*
#define NAN_DEBUG
*/

#define LOG_SUM_EXP_LOOKUP_MAX 10
#define LOG_SUM_EXP_LOOKUP_PRECISION .0001

#define LOG_SUM_EXP_LOOKUP_ENTRIES (((int) (LOG_SUM_EXP_LOOKUP_MAX / LOG_SUM_EXP_LOOKUP_PRECISION)) + 1)

/* comment to disable interpolation */
#define LOG_SUM_EXP_INTERPOLATE

typedef double LogProb;

double log_sum_exp_unary_slow (double x);  /* does not use lookup table */

struct LogSumExpLookupTable {
  double *lookup;
  LogSumExpLookupTable();
  ~LogSumExpLookupTable();
};

extern LogSumExpLookupTable logSumExpLookupTable;

inline double log_sum_exp_unary (double x) {
  /* returns log(1 + exp(-x)) for nonnegative x */
#ifdef LOG_SUM_EXP_SLOW
  return log_sum_exp_unary_slow(x);
#else /* LOG_SUM_EXP_SLOW */
  if (x >= LOG_SUM_EXP_LOOKUP_MAX || std::isnan(x) || std::isinf(x))
    return 0;
  if (x < 0) {  /* really dumb approximation for x < 0. Should never be encountered, so issue a warning */
    cerr << "Called log_sum_exp_unary(x) for negative x = " << x << endl;
    return -x;
  }
  const int n = (int) (x / LOG_SUM_EXP_LOOKUP_PRECISION);
  const double f0 = logSumExpLookupTable.lookup[n];
#ifdef LOG_SUM_EXP_INTERPOLATE
  const double dx = x - (n * LOG_SUM_EXP_LOOKUP_PRECISION);
  const double f1 = logSumExpLookupTable.lookup[n+1];
  const double df = f1 - f0;
  return f0 + df * (dx / LOG_SUM_EXP_LOOKUP_PRECISION);
#else /* LOG_SUM_EXP_INTERPOLATE */
  return f0;
#endif /* LOG_SUM_EXP_INTERPOLATE */
#endif /* LOG_SUM_EXP_SLOW */
}

inline double log_sum_exp (double a, double b) {
  /* returns log(exp(a) + exp(b)) */
  double max, diff, ret;
  // Note: Infinity plus or minus a finite quantity is still Infinity,
  // but Infinity - Infinity = NaN.
  // Thus, we are susceptible to NaN errors when trying to add 0+0 in log-space.
  // To work around this, we explicitly test for a==b.
  if (a == b) { max = a; diff = 0; }
  else if (a < b) { max = b; diff = b - a; }
  else { max = a; diff = a - b; }
  ret = max + log_sum_exp_unary (diff);
#if defined(NAN_DEBUG)
  if (std::isnan(ret)) {
    cerr << "NaN error in log_sum_exp" << endl;
    throw;
  }
#endif
  return ret;
}

inline double log_sum_exp (double a, double b, double c) {
    return log_sum_exp (log_sum_exp (a, b), c);
}

inline double log_sum_exp (double a, double b, double c, double d) {
    return log_sum_exp (log_sum_exp (log_sum_exp (a, b), c), d);
}

inline void log_accum_exp (double& a, double b) {
  a = log_sum_exp (a, b);
}

inline double log_sum_exp (double a, double b, double c, double d, double e) {
    return log_sum_exp (log_sum_exp (log_sum_exp (log_sum_exp (a, b), c), d), e);
}

inline double log_sum_exp (const vguard<double>& v) {
  double lpTot = -numeric_limits<double>::infinity();
  for (auto lp : v)
    log_accum_exp (lpTot, lp);
  return lpTot;
}

inline double log_sum_exp (const vguard<vguard<double> >& v) {
  double lpTot = -numeric_limits<double>::infinity();
  for (auto lp : v)
    log_accum_exp (lpTot, log_sum_exp (lp));
  return lpTot;
}

double log_sum_exp_slow (double a, double b);  /* does not use lookup table */
double log_sum_exp_slow (double a, double b, double c);
double log_sum_exp_slow (double a, double b, double c, double d);

void log_accum_exp_slow (double& a, double b);

vguard<LogProb> log_vector (const vguard<double>& v);

vguard<LogProb> log_gsl_vector (gsl_vector* v);
vguard<double> gsl_vector_to_stl (gsl_vector* v);

vguard<vguard<LogProb> > log_vector_gsl_vector (const vguard<gsl_vector*>& v);

vguard<vguard<double> > gsl_matrix_to_stl (gsl_matrix* m);
gsl_matrix* stl_to_gsl_matrix (const vguard<vguard<double> >& m);

inline LogProb logInnerProduct (const vguard<LogProb>& v1, const vguard<LogProb>& v2) {
  LogProb lip = -numeric_limits<double>::infinity();
  for (vguard<LogProb>::const_iterator iter1 = v1.begin(), iter2 = v2.begin(); iter1 != v1.end(); ++iter1, ++iter2)
    lip = log_sum_exp (lip, *iter1 + *iter2);
  return lip;
}

inline LogProb logInnerProduct (const vguard<LogProb>& v1, const vguard<LogProb>& v2, const vguard<LogProb>& v3) {
  LogProb lip = -numeric_limits<double>::infinity();
  for (vguard<LogProb>::const_iterator iter1 = v1.begin(), iter2 = v2.begin(), iter3 = v3.begin(); iter1 != v1.end(); ++iter1, ++iter2, ++iter3)
    lip = log_sum_exp (lip, *iter1 + *iter2 + *iter3);
  return lip;
}

inline LogProb logInnerProduct (const vguard<vguard<LogProb> >& v1, const vguard<vguard<LogProb> >& v2) {
  LogProb lip = -numeric_limits<double>::infinity();
  for (vguard<vguard<LogProb> >::const_iterator iter1 = v1.begin(), iter2 = v2.begin(); iter1 != v1.end(); ++iter1, ++iter2)
    lip = log_sum_exp (lip, logInnerProduct (*iter1, *iter2));
  return lip;
}

double logBetaPdf (double prob, double yesCount, double noCount);
double logGammaPdf (double rate, double eventCount, double waitTime);
double logDirichletPdf (const vguard<double>& prob, const vguard<double>& count);

#endif /* LOGSUMEXP_INCLUDED */
