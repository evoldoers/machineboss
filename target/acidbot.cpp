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

  try {
    
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help,h", "display this help message")
      ("load,l", po::value<vector<string> >(), "load machine from JSON file")
      ("save,s", po::value<string>(), "save machine to JSON file")
      ("verbose,v", po::value<int>()->default_value(2), "verbosity level")
      ("log", po::value<vector<string> >(), "log everything in this function")
      ("nocolor", "log in monochrome")
      ;

    po::positional_options_description p;
    p.add("load", -1);

    po::variables_map vm;
    po::store (po::command_line_parser(argc,argv).options(desc).positional(p).run(), vm);
    po::notify(vm);    

    // parse args
    if (vm.count("help")) {
      cout << desc << "\n";
      return 1;
    }

    logger.parseLogArgs (vm);

    // load transducers
    if (!vm.count("load"))
      throw runtime_error ("Please load at least one machine");

    const vector<string> machines = vm.at("load").as<vector<string> >();
    Machine machine;
    int n = 0;
    for (auto iter = machines.rbegin(); iter != machines.rend(); ++iter) {
      LogThisAt(2,(n ? "Pre-composing" : "Starting") << " with transducer " << *iter << endl);
      const char* filename = (*iter).c_str();
      if (n++)
	machine = Machine::compose (MachineLoader::fromFile(filename), machine);
      else
	machine = MachineLoader::fromFile(filename);
    }

    // save transducer
    if (vm.count("save")) {
      const string savefile = vm.at("save").as<string>();
      ofstream out (savefile);
      machine.writeJson (out);
    } else
      machine.writeJson (cout);

  } catch (const std::exception& e) {
    cerr << e.what() << endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}
