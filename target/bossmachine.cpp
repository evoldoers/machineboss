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
#include "../src/schema.h"

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
      ;

    po::options_description createOpts("Transducer construction");
    createOpts.add_options()
      ("load,l", po::value<string>(), "load machine from file")
      ("preset,p", po::value<string>(), (string ("select preset (") + join (MachinePresets::presetNames(), ", ") + ")").c_str())
      ("generate,g", po::value<string>(), "sequence generator '<'")
      ("accept,a", po::value<string>(), "sequence acceptor '>'")
      ("null,n", "null transducer")
      ("weight,w", po::value<string>(), "weighted null transition '#'")
      ;

    po::options_description prefixOpts("Prefix operations");
    prefixOpts.add_options()
      ("reverse,e", "reverse")
      ("revcomp,r", "reverse-complement '~'")
      ("flip,f", "flip input/output")
      ;

    po::options_description postfixOpts("Postfix operations");
    postfixOpts.add_options()
      ("zero-or-one,z", "union with null '?'")
      ("kleene-star,k", "Kleene star '*'")
      ("kleene-plus,K", "Kleene plus '+'")
      ;

    po::options_description infixOpts("Infix operations");
    infixOpts.add_options()
      ("compose,m", "compose '=>'")
      ("concat,c", "concatenate '.'")
      ("and,i", "intersect '&&'")
      ("or,u", "union '||'")
      ("loop,o", "loop: x '?+' y = x(y.x)*")
      ;

    po::options_description miscOpts("Miscellaneous");
    miscOpts.add_options()
      ("begin,B", "left bracket '('")
      ("end,E", "right bracket ')'")
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

    po::options_description transOpts("");
    transOpts.add(createOpts).add(prefixOpts).add(postfixOpts).add(infixOpts).add(miscOpts);

    po::options_description helpOpts("");
    helpOpts.add(generalOpts).add(createOpts).add(prefixOpts).add(postfixOpts).add(infixOpts).add(miscOpts).add(appOpts);

    po::options_description parseOpts("");
    parseOpts.add(generalOpts).add(appOpts);

    map<string,string> alias;
    alias[string("<")] = "--generate";
    alias[string(">")] = "--accept";
    alias[string("=>")] = "--compose";
    alias[string(".")] = "--concat";
    alias[string("&&")] = "--and";
    alias[string("||")] = "--or";
    alias[string("?")] = "--zero-or-one";
    alias[string("*")] = "--kleene-star";
    alias[string("+")] = "--kleene-plus";
    alias[string("?+")] = "--loop";
    alias[string("#")] = "--weight";
    alias[string("~")] = "--revcomp";
    alias[string("(")] = "--begin";
    alias[string(")")] = "--end";

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
	  if (machines.empty() || !lastCommand.empty()) {
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
	  if (!desc && aliasedArg.size() > 1 && aliasedArg[1] == '-')
	    desc = transOpts.find_nothrow (aliasedArg.substr(2), false);
	  if (desc)
	    LogThisAt(3,"Command '" << arg << "' ==> " << desc->description() << endl);
	  else
	    LogThisAt(3,"Warning: unrecognized command '" << arg << "'" << endl);
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
	} else if (command == "--compose")
	  m = Machine::compose (popMachine(), nextMachine());
	else if (command == "--concat")
	  m = Machine::concatenate (popMachine(), nextMachine());
	else if (command == "--and")
	  m = Machine::intersect (popMachine(), nextMachine());
	else if (command == "--or")
	  m = Machine::takeUnion (popMachine(), nextMachine());
	else if (command == "--zero-or-one")
	  m = Machine::zeroOrOne (popMachine());
	else if (command == "--kleene-star")
	  m = Machine::kleeneStar (popMachine());
	else if (command == "--kleene-plus")
	  m = Machine::kleenePlus (popMachine());
	else if (command == "--loop")
	  m = Machine::kleeneLoop (popMachine(), nextMachine());
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
	else if (command == "--weight") {
	  const string wArg = getArg();
	  WeightExpr w;
	  try {
	    w = json::parse(wArg);
	    if (!MachineSchema::validate ("expr", w))
	      w = WeightExpr(wArg);
	  } catch (...) {
	    w = WeightExpr(wArg);
	  }
	  m = Machine::singleTransition (w);
	} else if (command == "--begin") {
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
