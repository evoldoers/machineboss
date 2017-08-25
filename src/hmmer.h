#ifndef HMMER_INCLUDED
#define HMMER_INCLUDED

#include <fstream>
#include "vguard.h"
#include "machine.h"

struct HmmerModel {
  struct Node {
    vguard<double> matchEmit, insEmit;
    double m_to_m, m_to_i, m_to_d, i_to_m, i_to_i, d_to_m, d_to_d;
  };
  double b_to_m1, b_to_i0, b_to_d1, i0_to_m1, i0_to_i0;
  vguard<Node> node;

  HmmerModel() { }
  void read (ifstream&);
  static double strToProb (const string&);
  static vguard<double> strsToProbs (const vguard<string>&);
  
  Machine machine() const;
};

#endif /* HMMER_INCLUDED */
