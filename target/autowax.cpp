#include <time.h>
#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <random>
#include <deque>
#include <random>
#include <regex>
#include <boost/program_options.hpp>

#include "../src/logger.h"
#include "../src/assembly.h"

using namespace std;
namespace po = boost::program_options;

int main (int argc, char** argv) {

  try {

    // Declare the supported options.
    po::options_description generalOpts("General options");
    generalOpts.add_options()
      ("help,h", "display this help message")
      ("verbose,v", po::value<int>()->default_value(2), "verbosity level")
      ("debug,d", po::value<vector<string> >(), "log specified function")
      ("monochrome,b", "log in black & white")
      ("seed", po::value<int>(), "random number seed")
      ;

    po::options_description polishOpts("Polishing options");
    polishOpts.add_options()
      ("assembly,a", po::value<string>()->required(), "load assembly data from JSON file")
      ;

    po::options_description opts("");
    opts.add(generalOpts).add(polishOpts);

    // parse options
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc,argv).options(opts).run();
    po::store (parsed, vm);
    
    // deal with help
    if (vm.count("help")) {
      cout << opts << endl;
      return 1;
    }
    logger.parseLogArgs (vm);

    // check required options are present
    po::notify(vm);    

    // random seed
    auto makeRnd = [&] () -> mt19937 {
      time_t timer;
      time (&timer);
      const int seed = vm.count("seed") ? vm.at("seed").as<int>() : timer;
      LogThisAt(2,"Random seed is " << seed << endl);
      mt19937 mt (seed);
      return mt;
    };

    // load & validate assembly
    const Assembly assembly = JsonLoader<Assembly>::fromFile (vm.at("assembly").as<string>());

    // write assembly
    assembly.writeJson (cout);
    
  } catch (const std::exception& e) {
    cerr << e.what() << endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}
