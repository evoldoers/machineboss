#include <iostream>
#include <gsl/gsl_randist.h>
#include "logsumexp.h"
#include "util.h"

LogSumExpLookupTable logSumExpLookupTable = LogSumExpLookupTable();

LogSumExpLookupTable::LogSumExpLookupTable() {
  lookup = new double [LOG_SUM_EXP_LOOKUP_ENTRIES];
  int n;
  double x;
  for (n = 0; n < LOG_SUM_EXP_LOOKUP_ENTRIES; ++n) {
    x = n * LOG_SUM_EXP_LOOKUP_PRECISION;
    lookup[n] = log_sum_exp_unary_slow(x);
  }
}

LogSumExpLookupTable::~LogSumExpLookupTable() {
  delete[] lookup;
}

double log_sum_exp_slow (double a, double b) {
  double min, max, diff, ret;
  if (a < b) { min = a; max = b; }
  else { min = b; max = a; }
  if (min == -numeric_limits<double>::infinity())
    return max;
  diff = max - min;
  ret = max + log_sum_exp_unary_slow (diff);
#if defined(NAN_DEBUG)
  if (std::isnan(ret)) {
    cerr << "NaN error in log_sum_exp" << endl;
    throw;
  }
#endif
  return ret;
}

double log_sum_exp_slow (double a, double b, double c) {
  return log_sum_exp_slow (log_sum_exp_slow (a, b), c);
}

double log_sum_exp_slow (double a, double b, double c, double d) {
  return log_sum_exp_slow (log_sum_exp_slow (log_sum_exp_slow (a, b), c), d);
}

double log_sum_exp_unary_slow (double x) {
  return log (1. + exp(-x));
}

void log_accum_exp_slow (double& a, double b) {
  a = log_sum_exp_slow (a, b);
}

vguard<LogProb> log_vector (const vguard<double>& v) {
  return transform_container<double,vguard<double> > (v, log);
}

vguard<LogProb> log_gsl_vector (gsl_vector* v) {
  vguard<LogProb> l (v->size);
  for (size_t i = 0; i < v->size; ++i)
    l[i] = log (gsl_vector_get (v, i));
  return l;
}

vguard<double> gsl_vector_to_stl (gsl_vector* v) {
  vguard<double> stlv (v->size);
  for (size_t i = 0; i < v->size; ++i)
    stlv[i] = gsl_vector_get (v, i);
  return stlv;
}

vguard<vguard<LogProb> > log_vector_gsl_vector (const vguard<gsl_vector*>& v) {
  vguard<vguard<LogProb> > result (v.size());
  for (size_t i = 0; i < v.size(); ++i)
    result[i] = log_gsl_vector (v[i]);
  return result;
}

vguard<vguard<double> > gsl_matrix_to_stl (gsl_matrix* m) {
  vguard<vguard<double> > vv (m->size1, vguard<double> (m->size2));
  for (size_t i = 0; i < m->size1; ++i)
    for (size_t j = 0; j < m->size2; ++j)
      vv[i][j] = gsl_matrix_get (m, i, j);
  return vv;
}

gsl_matrix* stl_to_gsl_matrix (const vguard<vguard<double> >& vv) {
  Assert (vv.size() > 0, "Matrix has no rows");
  Assert (vv.front().size() > 0, "Matrix has no columns");
  const size_t rows = vv.size(), cols = vv.front().size();
  gsl_matrix* m = gsl_matrix_alloc (rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    Assert (vv[i].size() == cols, "Matrix is uneven");
    for (size_t j = 0; j < cols; ++j)
      gsl_matrix_set (m, i, j, vv[i][j]);
  }
  return m;
}

double logBetaPdf (double prob, double yesCount, double noCount) {
  return log (gsl_ran_beta_pdf (prob, yesCount + 1, noCount + 1));
}

double logGammaPdf (double rate, double eventCount, double waitTime) {
  return log (gsl_ran_gamma_pdf (rate, eventCount + 1, 1. / waitTime));
}

double logDirichletPdf (const vguard<double>& prob, const vguard<double>& count) {
  Assert (prob.size() == count.size(), "Dimensionality of Dirichlet counts vguard does not match that of probability parameter vguard");
  vguard<double> countPlusOne (count);
  for (auto& c : countPlusOne)
    ++c;
  return log (gsl_ran_dirichlet_pdf (prob.size(), countPlusOne.data(), prob.data()));
}
