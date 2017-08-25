#include "hmmer.h"
#include "regexmacros.h"
#include "util.h"

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

void HmmerModel::read (ifstream& in) {
  node.clear();
  const regex tag_re ("^(" RE_PLUS(RE_CHAR_CLASS("A-Z")) ")");
  const regex end_re ("^//");
  string line;
  smatch tag_match;
  vguard<string> alph;
  while (getline(in,line))
    if (regex_match (line, tag_match, tag_re)) {
      const string tag = tag_match.str(1);
      if (tag == "HMM") {
	const vguard<string> hmm_alph = split (line);
	alph = vguard<string> (hmm_alph.begin() + 1, hmm_alph.end());
	for (int skip = 0; skip < 4; ++skip)
	  getline(in,line);
	const vguard<string> beginTrans = split (line);
	b_to_m1 = strToProb (beginTrans[0]);
	b_to_i0 = strToProb (beginTrans[1]);
	b_to_d1 = strToProb (beginTrans[2]);
	i0_to_m1 = strToProb (beginTrans[3]);
	i0_to_i0 = strToProb (beginTrans[4]);
	while (getline(in,line)) {
	  if (regex_match (line, end_re))
	    break;
	  vguard<string> nodeMatchEmit = split (line);
	  Assert (nodeMatchEmit.size() == alph.size() + 6, "HMM parse error: wrong number of fields in node match line");
	  Assert (stoi(nodeMatchEmit[0]) == node.size() + 1, "HMM parse error: incorrect node index");
	  Assert (getline(in,line), "HMM parse error: premature truncation of node after match line");
	  const vguard<string> nodeInsEmit = split (line);
	  Assert (nodeMatchEmit.size() == alph.size(), "HMM parse error: wrong number of fields in node insert line");
	  Assert (getline(in,line), "HMM parse error: premature truncation of node after insert line");
	  const vguard<string> nodeTrans = split (line);
	  Assert (nodeTrans.size() == 7, "HMM parse error: wrong number of fields in node transitions line");
	  Node n;
	  nodeMatchEmit.erase (nodeMatchEmit.begin());
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

Machine HmmerModel::machine() const {
  Machine m;
  // TODO: write me
  return m;
}

