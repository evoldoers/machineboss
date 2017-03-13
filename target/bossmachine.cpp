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
      ("log,L", po::value<vector<string> >(), "log specified function")
      ("nocolor,N", "log in monochrome")
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
      ("load,d", po::value<string>(), "load machine from file")
      ("preset,t", po::value<string>(), (string ("preset machine (") + join (MachinePresets::presetNames(), ", ") + ")").c_str())
      ("generate,g", po::value<string>(), "sequence generator")
      ("accept,a", po::value<string>(), "sequence acceptor")
      ("pipe,p", po::value<string>(), "pipe (compose) machine")
      ("and,i", po::value<string>(), "intersect machine '&'")
      ("or,u", po::value<string>(), "take union with machine '|'")
      ("concat,c", po::value<string>(), "concatenate machine '+'")
      ("kleene,k", "Kleene closure (postfix)")
      ("kleene-weight,K", po::value<string>(), "weighted Kleene closure")
      ("kleene-loop,l", po::value<string>(), "Kleene closure via loop machine")
      ("reverse,e", po::value<string>(), "reverse")
      ("revcomp,r", po::value<string>(), "reverse-complement")
      ("flip,f", po::value<string>(), "flip input/output")
      ("null,n", "null transducer")
      ("weight,w", po::value<string>(), "single weighted transition")
      ("begin,B", "left bracket '('")
      ("end,E", "right bracket ')'")
      ;

    map<string,string> alias;
    alias[string("(")] = "--begin";
    alias[string(")")] = "--end";
    alias[string("&")] = "--and";
    alias[string("|")] = "--or";
    alias[string("+")] = "--concat";
    alias[string("#")] = "--weight";

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
    auto reduceMachines = [&]() -> Machine {
      Machine machine = machines.back();
      do {
	machines.pop_back();
	if (machines.size())
	  machine = Machine::compose (machines.back(), machine);
      } while (machines.size());
      return machine;
    };

    const vector<string> argVec = po::collect_unrecognized (parsed.options, po::include_positional);
    deque<string> args (argVec.begin(), argVec.end());
    while (!args.empty()) {
      function<Machine(const string&)> nextMachineForCommand;
      auto pushNextMachine = [&]() {
	machines.push_back (nextMachineForCommand (string()));
      };
      nextMachineForCommand = [&] (const string& lastCommand) -> Machine {
	if (args.empty()) {
	  cout << helpOpts << endl;
	  throw runtime_error (lastCommand.size() ? (string("Missing argument for ") + lastCommand) : string("Missing command"));
	}
	const string arg = args.front();
	args.pop_front();
	auto getArg = [&] () -> string {
	  if (args.empty()) {
	    cout << helpOpts << endl;
	    throw runtime_error (string("Missing argument for ") + arg);
	  }
	  const string arg = args.front();
	  args.pop_front();
	  return arg;
	};
	auto popMachine = [&] () -> Machine {
	  if (machines.empty()) {
	    cout << helpOpts << endl;
	    throw runtime_error (string("Missing machine for ") + arg);
	  }
	  const Machine m = machines.back();
	  machines.pop_back();
	  return m;
	};
	auto nextMachine = [&] () -> Machine {
	  return nextMachineForCommand (arg);
	};

	const string aliasedArg = alias.count(arg) ? alias.at(arg) : arg;
	const po::option_description* desc = NULL;
	if (aliasedArg[0] == '-') {
	  desc = transOpts.find_nothrow (aliasedArg, false);
	  if (desc)
	    LogThisAt(3,"Command " << arg << " ==> " << desc->description() << endl);
	  else
	    LogThisAt(3,"Warning: unrecognized command " << arg << endl);
	}
	const string command = desc ? (string("--") + desc->long_name()) : arg;

	Machine m;
	if (command[0] != '-')
	  m = MachineLoader::fromFile (command);
	else if (command == "--load")
	  m = MachineLoader::fromFile (getArg());
	else if (command == "--preset")
	  m = MachinePresets::makePreset (getArg().c_str());
	else if (command == "--generate") {
	  const NamedInputSeq inSeq = NamedInputSeq::fromFile (getArg());
	  m = Machine::generator (inSeq.name, inSeq.seq);
	} else if (command == "--accept") {
	  const NamedOutputSeq outSeq = NamedOutputSeq::fromFile (getArg());
	  m = Machine::acceptor (outSeq.name, outSeq.seq);
	} else if (command == "--pipe")
	  m = Machine::compose (popMachine(), nextMachine());
	else if (command == "--and")
	  m = Machine::intersect (popMachine(), nextMachine());
	else if (command == "--or")
	  m = Machine::takeUnion (popMachine(), nextMachine());
	else if (command == "--concat")
	  m = Machine::concatenate (popMachine(), nextMachine());
	else if (command == "--union")
	  m = Machine::takeUnion (popMachine(), popMachine());
	else if (command == "--kleene")
	  m = popMachine().kleeneClosure();
	else if (command == "--kleene-weight")
	  m = popMachine().kleeneClosure (getArg());
	else if (command == "--kleene-loop")
	  m = popMachine().kleeneClosure (nextMachine());
	else if (command == "--reverse")
	  m = nextMachine().reverse();
	else if (command == "--revcomp") {
	  const Machine r = nextMachine();
	  const vguard<OutputSymbol> outAlph = m.outputAlphabet();
	  const set<OutputSymbol> outAlphSet (outAlph.begin(), outAlph.end());
	  m = Machine::compose (r.reverse(),
				MachinePresets::makePreset ((outAlphSet.count(string("U")) || outAlphSet.count(string("u")))
							    ? "comprna"
							    : "compdna"));
	} else if (command == "--flip")
	  m = nextMachine().flipInOut();
	else if (command == "--null")
	  m = Machine::null();
	else if (command == "--weight")
	  m = Machine::singleTransition (getArg());
	else if (command == "--begin") {
	  list<Machine> pushedMachines;
	  swap (pushedMachines, machines);
	  while (true) {
	    if (args.empty())
	      throw runtime_error (string("Unmatched '") + arg + "'");
	    const string& nextArg = args.front();
	    if (nextArg == "--end" || nextArg == "-E" || nextArg == ")")
	      break;
	    pushNextMachine();
	  }
	  const string endArg = getArg();
	  if (machines.empty())
	    throw runtime_error (string("Empty '") + arg + "' ... '" + endArg + "'");
	  m = reduceMachines();
	  swap (pushedMachines, machines);
	} else if (command == "--end")
	  throw runtime_error (string("Unmatched '") + arg + "'");
	else {
	  cout << helpOpts << endl;
	  throw runtime_error (string ("Unknown option: ") + arg);
	}
	return m;
      };
      pushNextMachine();
    }

    // compose remaining transducers
    if (machines.empty()) {
      cout << helpOpts << endl;
      cout << "Please specify a transducer" << endl;
      return 1;
    }
    const Machine machine = reduceMachines();
    
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
