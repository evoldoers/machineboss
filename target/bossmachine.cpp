#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <random>
#include <deque>
#include <boost/program_options.hpp>

#include "../src/vguard.h"
#include "../src/logger.h"
#include "../src/fastseq.h"
#include "../src/machine.h"
#include "../src/preset.h"
#include "../src/seqpair.h"
#include "../src/constraints.h"
#include "../src/params.h"
#include "../src/fitter.h"
#include "../src/viterbi.h"

using namespace std;

namespace po = boost::program_options;

int main (int argc, char** argv) {

  try {

    // Declare the supported options.
    po::options_description generalOpts("General options");
    generalOpts.add_options()
      ("help,h", "display this help message")
      ("verbose,v", po::value<int>()->default_value(2), "verbosity level")
      ("log", po::value<vector<string> >(), "log specified function")
      ("nocolor", "log in monochrome")
      ;

    po::options_description appOpts("Transducer application");
    appOpts.add_options()
      ("save,S", po::value<string>(), "save machine")
      ("params,P", po::value<string>(), "load parameter file")
      ("constraints,C", po::value<string>(), "load constraints file")
      ("data,D", po::value<string>(), "load sequence-pairs file")
      ("fit,F", "Baum-Welch parameter fit")
      ("align,A", "Viterbi sequence alignment")
      ;

    po::options_description transOpts("Transducer manipulation");
    transOpts.add_options()
      ("generate,g", po::value<string>(), "sequence generator")
      ("accept,a", po::value<string>(), "sequence acceptor")
      ("pipe,p", po::value<string>(), "pipe (compose) machine")
      ("compose", "compose last two machines")
      ("concat,c", po::value<string>(), "concatenate machine")
      ("append", "concatenate last two machines")
      ("or", po::value<string>(), "take union with machine")
      ("union,u", "union of last two machines")
      ("weight,w", po::value<string>(), "weighted union of last two machines")
      ("kleene,k", "Kleene closure")
      ("loop,l", po::value<string>(), "weighted Kleene closure")
      ("reverse,R", "reverse")
      ("revcomp,r", "reverse complement")
      ("flip,f", "flip input/output")
      ("null,n", "null transducer")
      ;

    po::options_description helpOpts("");
    helpOpts.add(generalOpts).add(transOpts).add(appOpts);

    po::options_description parseOpts("");
    parseOpts.add(generalOpts).add(appOpts);

    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc,argv).options(parseOpts).allow_unregistered().run();
    po::store (parsed, vm);
    po::notify(vm);    
      
    // parse args
    if (vm.count("help")) {
      cout << helpOpts << endl;
      return 1;
    }
    logger.parseLogArgs (vm);

    // create transducer
    list<Machine> machines;
    const vector<string> commandVec = po::collect_unrecognized (parsed.options, po::include_positional);
    deque<string> commands (commandVec.begin(), commandVec.end());
    while (!commands.empty()) {
      const string command = commands.front();
      commands.pop_front();
      auto getArg = [&] () -> string {
	if (commands.empty()) {
	  cout << helpOpts << endl;
	  throw runtime_error (string("Missing argument for ") + command);
	}
	const string arg = commands.front();
	commands.pop_front();
	return arg;
      };
      auto getMachine = [&] () -> Machine {
	if (machines.empty()) {
	  cout << helpOpts << endl;
	  throw runtime_error (string("Missing machine for ") + command);
	}
	const Machine m = machines.back();
	machines.pop_back();
	return m;
      };
      if (command[0] != '-')
	machines.push_back (MachineLoader::fromFile (command));
      else if (command == "--load")
	machines.push_back (MachineLoader::fromFile (getArg()));
      else if (command == "--compose")
	machines.push_back (Machine::compose (getMachine(), getMachine()));
      else if (command == "--append")
	machines.push_back (Machine::concatenate (getMachine(), getMachine()));
      else if (command == "--concat" || command == "-c")
	machines.push_back (Machine::concatenate (getMachine(), MachineLoader::fromFile (getArg())));
      else if (command == "--pipe" || command == "-p")
	machines.push_back (Machine::compose (getMachine(), MachineLoader::fromFile (getArg())));
      else if (command == "--generate" || command == "-g") {
	const NamedInputSeq inSeq = NamedInputSeq::fromFile (getArg());
	machines.push_back (Machine::generator (inSeq.name, inSeq.seq));
      } else if (command == "--accept" || command == "-a") {
	const NamedOutputSeq outSeq = NamedOutputSeq::fromFile (getArg());
	machines.push_back (Machine::acceptor (outSeq.name, outSeq.seq));
      } else if (command == "--union" || command == "-u")
	machines.push_back (Machine::unionOf (getMachine(), getMachine()));
      else if (command == "--weighted-union" || command == "-w")
	machines.push_back (Machine::unionOf (getMachine(), getMachine(), getArg()));
      else if (command == "--or")
	machines.push_back (Machine::unionOf (getMachine(), MachineLoader::fromFile (getArg())));
      else if (command == "--flip" || command == "-f")
	machines.push_back (getMachine().flipInOut());
      else if (command == "--reverse" || command == "-R")
	machines.push_back (getMachine().reverse());
      else if (command == "--revcomp" || command == "-r") {
	const Machine m = getMachine();
	const vguard<OutputSymbol> outAlph = m.outputAlphabet();
	const set<OutputSymbol> outAlphSet (outAlph.begin(), outAlph.end());
	machines.push_back (Machine::compose (m.reverse(),
					      MachinePresets::makePreset ((outAlphSet.count(string("U")) || outAlphSet.count(string("u")))
									  ? "comprna"
									  : "compdna")));
      } else if (command == "--kleene" || command == "-k")
	machines.push_back (getMachine().kleeneClosure());
      else if (command == "--loop" || command == "-l")
	machines.push_back (getMachine().kleeneClosure (getArg()));
      else if (command == "--null" || command == "-n")
	machines.push_back (Machine::null());
      else {
	cout << helpOpts << endl;
	cout << "Unknown option: " << command << endl;
	return 1;
      }
    }

    // compose remaining transducers
    if (machines.empty()) {
      cout << helpOpts << endl;
      cout << "Please specify a transducer" << endl;
      return 1;
    }
    
    Machine machine = machines.back();
    do {
      machines.pop_back();
      if (machines.size())
	machine = Machine::compose (machines.back(), machine);
    } while (machines.size());
    
    // save transducer
    if (vm.count("save")) {
      const string savefile = vm.at("save").as<string>();
      ofstream out (savefile);
      machine.writeJson (out);
    } else if (!vm.count("fit") && !vm.count("align"))
      machine.writeJson (cout);

    // do some syntax checking
    Require (!vm.count("params") || (vm.count("fit") || vm.count("align")), "Can't specify --params without --fit or --align");
    Require (!vm.count("data") || (vm.count("fit") || vm.count("align")), "Can't specify --data without --fit or --align");
    Require (!vm.count("constraints") || vm.count("fit"), "Can't specify --constraints without --fit");

    // fit parameters
    Params params;
    SeqPairList data;
    if (vm.count("fit")) {
      Require (vm.count("constraints") && vm.count("data"),
	       "To fit parameters, please specify a constraints file and a data file");
      MachineFitter fitter;
      fitter.machine = machine;
      fitter.constraints = Constraints::fromFile(vm.at("constraints").as<string>());
      fitter.seed = vm.count("params") ? Params::fromFile(vm.at("params").as<string>()) : fitter.constraints.defaultParams();
      data = SeqPairList::fromFile(vm.at("data").as<string>());
      params = fitter.fit(data);
      cout << params.toJsonString() << endl;
    }

    // align sequences
    if (vm.count("align")) {
      Require ((vm.count("data") && vm.count("params")) || vm.count("fit"),
	       "To align sequences, please specify a data file and a parameter file (or fit with --fit)");
      if (!vm.count("fit")) {
	params = Params::fromFile(vm.at("params").as<string>());
	data = SeqPairList::fromFile(vm.at("data").as<string>());
      }
      const EvaluatedMachine eval (machine, params);
      cout << "[";
      size_t n = 0;
      for (const auto& seqPair: data.seqPairs) {
	const ViterbiMatrix viterbi (eval, seqPair);
	const MachinePath path = viterbi.trace (machine);
	cout << (n++ ? ",\n " : "");
	path.writeJson (cout);
      }
      cout << "]\n";
    }
    
  } catch (const std::exception& e) {
    cerr << e.what() << endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}
