#include "hmmer.h"

void HmmerModel::read (ifstream& in) {
  node.clear();
  // TODO: write me
  const regex tag_re ("^" RE_PLUS(RE_CHAR_CLASS("A-Z")));
  string line;
  while (getline(in,line))
    if (regex_match (line, tag_re)) {
      
    }
}

Machine HmmerModel::machine() const {
  Machine m;
  // TODO: write me
  return m;
}

