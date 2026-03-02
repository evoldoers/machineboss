#ifndef HMMER_INCLUDED
#define HMMER_INCLUDED

#include <fstream>
#include "vguard.h"
#include "machine.h"

// HMMER3 model
namespace MachineBoss {

struct HmmerModel {
  struct Node {
    vguard<double> matchEmit, insEmit;
    double m_to_m, m_to_i, m_to_d, i_to_m, i_to_i, d_to_m, d_to_d;
  };
  double b_to_m1, b_to_i0, b_to_d1, i0_to_m1, i0_to_i0;
  vguard<double> ins0Emit;
  vguard<double> nullEmit;  // background emission probs (SwissProt composition)
  vguard<Node> node;
  vguard<string> alph;

  HmmerModel() { }
  void read (istream&);
  void loadNullModel();  // populate nullEmit from SwissProt background frequencies
  static double strToProb (const string&);
  static vguard<double> strsToProbs (const vguard<string>&);

  // Core state indices (used by machine() and plan7Machine())
  inline StateIndex b_idx() const { return 0; }
  inline StateIndex ix_idx (int n) const { return 5*n + 1; }
  inline StateIndex i_idx (int n) const { return 5*n + 2; }
  inline StateIndex mx_idx (int n) const { return 5*n - 2; }
  inline StateIndex m_idx (int n) const { return 5*n - 1; }
  inline StateIndex d_idx (int n) const { return 5*n; }
  inline StateIndex core_end_idx() const { return 5 * node.size() + 3; }
  inline StateIndex nCoreStates() const { return 5 * node.size() + 4; }

  // Plan7 flanking state indices (appended after core states)
  // In plan7Machine: B (index 0) is repurposed as S (start),
  // and plan7_b_idx() is the new Begin state with B's original transitions
  inline StateIndex n_idx() const { return nCoreStates(); }
  inline StateIndex nx_idx() const { return nCoreStates() + 1; }
  inline StateIndex plan7_b_idx() const { return nCoreStates() + 2; }
  inline StateIndex cx_idx() const { return nCoreStates() + 3; }
  inline StateIndex c_idx() const { return nCoreStates() + 4; }
  inline StateIndex jx_idx() const { return nCoreStates() + 5; }
  inline StateIndex j_idx() const { return nCoreStates() + 6; }
  inline StateIndex t_idx() const { return nCoreStates() + 7; }
  inline StateIndex nPlan7States() const { return nCoreStates() + 8; }

  vguard<double> calcMatchOccupancy() const;

  Machine machine (bool local = true) const;  // core-only (backward compat)
  Machine plan7Machine (bool multihit = false, double L = 400) const;  // full Plan7
};

}  // end namespace

#endif /* HMMER_INCLUDED */
