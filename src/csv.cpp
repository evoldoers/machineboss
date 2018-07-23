#include <algorithm>
#include "csv.h"
#include "fastseq.h"

Machine CSVProfile::machine() const {
  Machine m;
  m.state.resize (row.size() + 1);
  for (SeqIdx pos = 0; pos <= row.size(); ++pos)
    m.state[pos].name = to_string (pos);
  for (SeqIdx pos = 0; pos < row.size(); ++pos)
    for (size_t col = 0; col < row[pos].size() && col <= header.size(); ++col)
      m.state[pos].trans.push_back (MachineTransition (string(), col < header.size() ? header[col] : string(), pos + 1, WeightAlgebra::doubleConstant (row[pos][col])));
  return m;
}

void CSVProfile::readHeader (ifstream& in, const char* splitChars) {
  string line;
  if (getline (in, line))
    header = split (line, splitChars);
  header.erase (remove_if (header.begin(), header.end(), [](const string& s) { return s.empty(); }), header.end());
}

void CSVProfile::readRows (ifstream& in, const char* splitChars) {
  string line;
  while (getline (in, line)) {
    const vector<string> strCols = split (line, splitChars);
    if (strCols.size()) {
      vector<double> dblCols (strCols.size());
      for (size_t col = 0; col < strCols.size(); ++col)
	dblCols[col] = stof (strCols[col].c_str());
      row.push_back (dblCols);
    }
  }
}

void CSVProfile::read (ifstream& in, const char* splitChars) {
  readHeader (in, splitChars);
  readRows (in, splitChars);
}
