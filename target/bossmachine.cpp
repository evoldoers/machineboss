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
#include "../src/util.h"

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
      ("preset", po::value<string>(), (string ("preset transducer (") + join (MachinePresets::presetNames(), ", ") + ")").c_str())
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
      function<Machine(const string&)> nextMachine;
      nextMachine = [&] (const string& lastCommand) -> Machine {
	if (commands.empty()) {
	  cout << helpOpts << endl;
	  throw runtime_error (lastCommand.size() ? (string("Missing argument for ") + lastCommand) : string("Missing command"));
	}
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
	auto popMachine = [&] () -> Machine {
	  if (machines.empty()) {
	    cout << helpOpts << endl;
	    throw runtime_error (string("Missing machine for ") + command);
	  }
	  const Machine m = machines.back();
	  machines.pop_back();
	  return m;
	};
	Machine m;
	if (command[0] != '-')
	  m = MachineLoader::fromFile (command);
	else if (command == "--load")
	  m = MachineLoader::fromFile (getArg());
	else if (command == "--preset")   // undocumented...
	  m = MachinePresets::makePreset (getArg().c_str());
	else if (command == "--compose")
	  m = Machine::compose (popMachine(), popMachine());
	else if (command == "--append")
	  m = Machine::concatenate (popMachine(), popMachine());
	else if (command == "--concat" || command == "-c")
	  m = Machine::concatenate (popMachine(), nextMachine(command));
	else if (command == "--pipe" || command == "-p")
	  m = Machine::compose (popMachine(), nextMachine(command));
	else if (command == "--generate" || command == "-g") {
	  const NamedInputSeq inSeq = NamedInputSeq::fromFile (getArg());
	  m = Machine::generator (inSeq.name, inSeq.seq);
	} else if (command == "--accept" || command == "-a") {
	  const NamedOutputSeq outSeq = NamedOutputSeq::fromFile (getArg());
	  m = Machine::acceptor (outSeq.name, outSeq.seq);
	} else if (command == "--union" || command == "-u")
	  m = Machine::unionOf (popMachine(), popMachine());
	else if (command == "--weighted-union" || command == "-w")
	  m = Machine::unionOf (popMachine(), popMachine(), getArg());
	else if (command == "--or")
	  m = Machine::unionOf (popMachine(), nextMachine(command));
	else if (command == "--flip" || command == "-f")
	  m = popMachine().flipInOut();
	else if (command == "--reverse" || command == "-R")
	  m = popMachine().reverse();
	else if (command == "--revcomp" || command == "-r") {
	  const Machine r = popMachine();
	  const vguard<OutputSymbol> outAlph = m.outputAlphabet();
	  const set<OutputSymbol> outAlphSet (outAlph.begin(), outAlph.end());
	  m = Machine::compose (r.reverse(),
				MachinePresets::makePreset ((outAlphSet.count(string("U")) || outAlphSet.count(string("u")))
							    ? "comprna"
							    : "compdna"));
	} else if (command == "--kleene" || command == "-k")
	  m = popMachine().kleeneClosure();
	else if (command == "--loop" || command == "-l")
	  m = popMachine().kleeneClosure (getArg());
	else if (command == "--null" || command == "-n")
	  m = Machine::null();
	else {
	  cout << helpOpts << endl;
	  throw runtime_error (string ("Unknown option: ") + command);
	}
	return m;
      };
      machines.push_back (nextMachine(string()));
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
