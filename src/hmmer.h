#ifndef HMMER_INCLUDED
#define HMMER_INCLUDED

#include <fstream>
#include "vguard.h"
#include "machine.h"

// HMMER3 model
struct HmmerModel {
  struct Node {
    vguard<double> matchEmit, insEmit;
    double m_to_m, m_to_i, m_to_d, i_to_m, i_to_i, d_to_m, d_to_d;
  };
  double b_to_m1, b_to_i0, b_to_d1, i0_to_m1, i0_to_i0;
  vguard<double> ins0Emit;
  vguard<Node> node;
  vguard<string> alph;

  HmmerModel() { }
  void read (ifstream&);
  static double strToProb (const string&);
  static vguard<double> strsToProbs (const vguard<string>&);

  inline StateIndex b_idx() const { return 0; }
  inline StateIndex ix_idx (int n) const { return 5*n + 1; }
  inline StateIndex i_idx (int n) const { return 5*n + 2; }
  inline StateIndex mx_idx (int n) const { return 5*n - 2; }
  inline StateIndex m_idx (int n) const { return 5*n - 1; }
  inline StateIndex d_idx (int n) const { return 5*n; }
  inline StateIndex end_idx() const { return 5 * node.size() + 3; }
  inline StateIndex nStates() const { return 5 * node.size() + 4; }

  Machine machine() const;
};

#endif /* HMMER_INCLUDED */
