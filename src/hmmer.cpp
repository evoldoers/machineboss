#include <string>
#include "hmmer.h"
#include "regexmacros.h"
#include "util.h"

using namespace MachineBoss;

double HmmerModel::strToProb (const string& s) {
  return s == "*" ? 0 : exp(-stof(s));
}

vguard<double> HmmerModel::strsToProbs (const vguard<string>& s) {
  vguard<double> p;
  p.reserve (s.size());
  for (auto& str: s)
    p.push_back (strToProb (str));
  return p;
}

void HmmerModel::read (istream& in) {
  node.clear();
  const regex tag_re ("^(" RE_PLUS(RE_CHAR_CLASS("A-Z")) ")");
  const regex end_re ("^//");
  string line;
  smatch tag_match;
  while (getline(in,line))
    if (regex_search (line, tag_match, tag_re)) {
      const string tag = tag_match.str(1);
      if (tag == "HMM") {
	const vguard<string> hmm_alph = split (line);
	Assert (hmm_alph.size() > 1, "HMM parse error: empty alphabet");
	alph = vguard<string> (hmm_alph.begin() + 1, hmm_alph.end());
	for (int skip = 0; skip < 3; ++skip)
	  if (!getline(in,line))
	    break;
	const vguard<string> ins0 = split (line);
	Assert (ins0.size() == alph.size(), "HMM parse error: wrong number of fields in node 0 insert line");
	ins0Emit = strsToProbs (ins0);
 	if (!getline(in,line))
	  break;
	const vguard<string> beginTrans = split (line);
	b_to_m1 = strToProb (beginTrans[0]);
	b_to_i0 = strToProb (beginTrans[1]);
	b_to_d1 = strToProb (beginTrans[2]);
	i0_to_m1 = strToProb (beginTrans[3]);
	i0_to_i0 = strToProb (beginTrans[4]);
	while (getline(in,line)) {
	  if (regex_match (line, end_re))
	    break;
	  const vguard<string> nodeMatchLine = split (line);
	  Assert (nodeMatchLine.size() == alph.size() + 6, "HMM parse error: wrong number of fields in node match line");
	  Assert (stoi(nodeMatchLine[0]) == node.size() + 1, "HMM parse error: incorrect node index");
	  Assert (getline(in,line), "HMM parse error: premature truncation of node after match line");
	  const vguard<string> nodeInsEmit = split (line);
	  Assert (nodeInsEmit.size() == alph.size(), "HMM parse error: wrong number of fields in node insert line");
	  Assert (getline(in,line), "HMM parse error: premature truncation of node after insert line");
	  const vguard<string> nodeTrans = split (line);
	  Assert (nodeTrans.size() == 7, "HMM parse error: wrong number of fields in node transitions line");
	  Node n;
	  const vguard<string> nodeMatchEmit (nodeMatchLine.begin() + 1, nodeMatchLine.begin() + alph.size() + 1);
	  n.matchEmit = strsToProbs (nodeMatchEmit);
	  n.insEmit = strsToProbs (nodeInsEmit);
	  n.m_to_m = strToProb (nodeTrans[0]);
	  n.m_to_i = strToProb (nodeTrans[1]);
	  n.m_to_d = strToProb (nodeTrans[2]);
	  n.i_to_m = strToProb (nodeTrans[3]);
	  n.i_to_i = strToProb (nodeTrans[4]);
	  n.d_to_m = strToProb (nodeTrans[5]);
	  n.d_to_d = strToProb (nodeTrans[6]);
	  node.push_back (n);
	}
	break;
      }
    }
}

Machine HmmerModel::machine (bool local) const {
  Assert (node.size() > 0, "Attempt to create a transducer from an empty HMMER model");

  Machine m;
  m.state = vguard<MachineState> (nStates());

  m.state[b_idx()].name = "B";
  if (local) {
    // local mode entry probabilities from p7_ProfileConfig() in HMMER3 source code
    const auto occ = calcMatchOccupancy();
    double Z = 0;
    for (int k = 1; k < node.size(); ++k)
      Z += occ[k] * (node.size() - k + 1);
    for (int k = 1; k < node.size(); ++k)
      m.state[b_idx()].trans.push_back (MachineTransition (string(), string(), m_idx(k), occ[k] / Z));
  } else {
    m.state[b_idx()].trans.push_back (MachineTransition (string(), string(), m_idx(1), b_to_m1));
    m.state[b_idx()].trans.push_back (MachineTransition (string(), string(), i_idx(0), b_to_i0));
    m.state[b_idx()].trans.push_back (MachineTransition (string(), string(), d_idx(1), b_to_d1));
  }

  m.state[ix_idx(0)].trans.push_back (MachineTransition (string(), string(), m_idx(1), i0_to_m1));
  m.state[ix_idx(0)].trans.push_back (MachineTransition (string(), string(), i_idx(0), i0_to_i0));

  for (size_t sym = 0; sym < alph.size(); ++sym)
    m.state[i_idx(0)].trans.push_back (MachineTransition (string(), alph[sym], ix_idx(0), ins0Emit[sym]));

  for (int n = 0; n <= node.size(); ++n) {
    const string ns = to_string(n);
    m.state[i_idx(n)].name = string("I") + ns;
    m.state[ix_idx(n)].name = string("Ix") + ns;
    if (n > 0) {
      m.state[m_idx(n)].name = string("M") + ns;
      m.state[mx_idx(n)].name = string("Mx") + ns;
      m.state[d_idx(n)].name = string("D") + ns;

      const bool end = (n == node.size());
      m.state[mx_idx(n)].trans.push_back (MachineTransition (string(), string(), end ? end_idx() : m_idx(n+1), node[n-1].m_to_m));
      m.state[mx_idx(n)].trans.push_back (MachineTransition (string(), string(), i_idx(n), node[n-1].m_to_i));
      if (!end)
	m.state[mx_idx(n)].trans.push_back (MachineTransition (string(), string(), d_idx(n+1), node[n-1].m_to_d));

      m.state[ix_idx(n)].trans.push_back (MachineTransition (string(), string(), end ? end_idx() : m_idx(n+1), node[n-1].i_to_m));
      m.state[ix_idx(n)].trans.push_back (MachineTransition (string(), string(), i_idx(n), node[n-1].i_to_i));

      m.state[d_idx(n)].trans.push_back (MachineTransition (string(), string(), end ? end_idx() : m_idx(n+1), node[n-1].d_to_m));
      if (!end)
	m.state[d_idx(n)].trans.push_back (MachineTransition (string(), string(), d_idx(n+1), node[n-1].d_to_d));

      for (size_t sym = 0; sym < alph.size(); ++sym) {
	m.state[m_idx(n)].trans.push_back (MachineTransition (string(), alph[sym], mx_idx(n), node[n-1].matchEmit[sym]));
	m.state[i_idx(n)].trans.push_back (MachineTransition (string(), alph[sym], ix_idx(n), node[n-1].insEmit[sym]));
      }
    }
  }
  m.state[end_idx()].name = "E";

  return m;
}

vguard<double> HmmerModel::calcMatchOccupancy() const {
  // Taken from p7_hmm_CalculateOccupancy() in HMMER3 source code:
  //   Calculates a vector <mocc[1..M]> containing probability
  //   that each match state is used in a sampled path through
  //   the model.
  vguard<double> mocc (node.size());
  mocc[0] = 0.;			               /* no M_0 state */
  mocc[1] = node[0].m_to_i + node[0].m_to_m;   /* initialize w/ 1 - B->D_1 */
  for (int k = 2; k < node.size(); k++)
    mocc[k] = mocc[k-1] * (node[k].m_to_m + node[k].m_to_i) + (1.0 - mocc[k-1]) * node[k].d_to_m;
  return mocc;
}
