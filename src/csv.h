#ifndef CSV_INCLUDED
#define CSV_INCLUDED

#include <fstream>
#include <vector>
#include "machine.h"

#define DefaultCSVSplitChars ","

// CSV profile
struct CSVProfile {
  vector<string> header;
  vector<vector<double> > row;
  CSVProfile() { }
  void readHeader (ifstream&, const char* splitChars = DefaultCSVSplitChars);
  void readRows (ifstream&, const char* splitChars = DefaultCSVSplitChars);
  void read (ifstream&, const char* splitChars = DefaultCSVSplitChars);

  Machine machine() const;
};

#endif /* HMMER_INCLUDED */
