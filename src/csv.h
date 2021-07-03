#ifndef CSV_INCLUDED
#define CSV_INCLUDED

#include <fstream>
#include <vector>
#include "machine.h"

#define DefaultCSVSplitChars ","

// CSV profile
namespace MachineBoss {

struct CSVProfile {
  vector<string> header;
  vector<vector<double> > row;
  CSVProfile() { }
  void readHeader (ifstream&, const char* splitChars = DefaultCSVSplitChars);
  void readRows (ifstream&, const char* splitChars = DefaultCSVSplitChars);
  void read (ifstream&, const char* splitChars = DefaultCSVSplitChars);

  Machine machine() const;
  Machine mergingMachine() const;  // merges consecutive repeated characters, as in Graves (2006) "Connectionist Temporal Classification"
};

}  // end namespace

#endif /* HMMER_INCLUDED */
