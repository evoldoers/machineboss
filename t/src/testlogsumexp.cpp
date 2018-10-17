#include <iostream>
#include <fstream>
#include <string.h>
#include "../src/logsumexp.h"

using namespace std;

int main (int argc, char **argv) {
  if (argc != 3 || (strcmp(argv[1],"-slow") != 0 && strcmp(argv[1],"-fast") != 0)) {
    cout << "Usage: " << argv[0] << " [-slow|-fast] <steps>\n";
    exit (EXIT_FAILURE);
  }

  const bool slow = strcmp(argv[1],"-slow") == 0;
  const double steps = atof (argv[2]);
  const double max = 2, step = max / steps;
  cerr << "(running in " << (slow ? "slow" : "fast") << " mode)" << endl;
  double total = 0;
  for (double x = 0; x <= max; x += step)
    total += (slow ? log_sum_exp_slow(x,max-x) : log_sum_exp(x,max-x));
  cout << "Total: " << total << endl;
  
  exit (EXIT_SUCCESS);
}
