#include <algorithm>
#include "csv.h"
#include "fastseq.h"
#include "eval.h"

Machine CSVProfile::machine() const {
  Machine m;
  m.state.resize (row.size() + 1);
  for (SeqIdx pos = 0; pos <= row.size(); ++pos)
    m.state[pos].name = to_string (pos);
  for (SeqIdx pos = 0; pos < row.size(); ++pos) {
    for (size_t col = 0; col < row[pos].size() && col <= header.size(); ++col)
      m.state[pos].trans.push_back (MachineTransition (string(), col < header.size() ? header[col] : string(), pos + 1, WeightAlgebra::doubleConstant (row[pos][col])));
  }
  return m;
}

Machine CSVProfile::mergingMachine() const {
  Assert (header.size(), "Need header to build mergingMachine from CSVProfile");
  const OutputToken nCols = header.size();
  const SeqIdx nRows = row.size();
  Machine m;
  auto getStateIndex = [&] (SeqIdx pos, OutputToken lastTok) -> StateIndex {
    return pos == 0 ? 0 : ((pos - 1) * (nCols + 1) + (pos == nRows ? 0 : lastTok) + 1);
  };
  m.state.resize (getStateIndex (nRows, 0) + 1);
  for (SeqIdx pos = 1; pos < row.size(); ++pos)
    for (size_t tok = 0; tok <= nCols; ++tok)
      m.state[getStateIndex(pos,tok)].name = json::array ({{ pos, (tok == nCols ? string() : header[tok]) }});
  m.state.front().name = MachineStartTag;
  m.state.back().name = MachineEndTag;
  for (SeqIdx pos = 0; pos < row.size(); ++pos) {
    for (size_t col = 0; col < row[pos].size() && col <= header.size(); ++col) {
      const StateIndex dest = getStateIndex (pos + 1, col);
      const WeightExpr weight = WeightAlgebra::doubleConstant (row[pos][col]);
      for (size_t tok = 0; tok <= (pos ? nCols : 0); ++tok) {
	const StateIndex src = getStateIndex (pos, tok);
	const string emit = ((col == tok && pos > 0) || col == header.size()) ? string() : header[col];
	m.state[src].trans.push_back (MachineTransition (string(), emit, dest, weight));
      }
    }
  }
  return m;
}

void CSVProfile::readHeader (ifstream& in, const char* splitChars) {
  string line;
  if (getline (in, line))
    header = split (line, splitChars);
  while (header.size() && header.back().empty())
    header.pop_back();
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
