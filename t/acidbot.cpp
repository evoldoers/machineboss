#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <random>
#include <boost/program_options.hpp>

#include "../src/vguard.h"
#include "../src/logger.h"
#include "../src/fastseq.h"
#include "../src/trans.h"

using namespace std;

namespace po = boost::program_options;

int main (int argc, char** argv) {

#ifndef DEBUG
  try {
#endif /* DEBUG */
    
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help,h", "display this help message")
      ("compose,c", po::value<vector<string> >(), "load machine from JSON file")
      ("save,s", po::value<string>(), "save machine to JSON file")
      ("verbose,v", po::value<int>()->default_value(2), "verbosity level")
      ("log", po::value<vector<string> >(), "log everything in this function")
      ("nocolor", "log in monochrome")
      ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    // parse args
    if (vm.count("help")) {
      cout << desc << "\n";
      return 1;
    }

    logger.parseLogArgs (vm);

    // load transducers
    if (!vm.count("compose"))
      throw runtime_error ("Please specify at least one machine");

    const vector<string> comps = vm.at("compose").as<vector<string> >();
    Machine machine;
    bool first = true;
    for (auto iter = comps.rbegin(); iter != comps.rend(); ++iter) {
      const Machine mc = Machine::fromFile((*iter).c_str());
      if (first) {
	LogThisAt(3,"Loading " << *iter << endl);
	machine = mc;
	first = false;
      } else {
	LogThisAt(3,"Pre-composing with " << *iter << endl);
	machine = Machine::compose (mc, machine);
      }
    }

    // save transducer
    if (vm.count("save")) {
      const string savefile = vm.at("save").as<string>();
      if (savefile == "-")
	machine.writeJson (cout);
      else {
	  ofstream out (savefile);
	  machine.writeJson (out);
      }
    } else
      machine.writeJson (cout);

#ifndef DEBUG
  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }
#endif /* DEBUG */
  
  return EXIT_SUCCESS;
}
