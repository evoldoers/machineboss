#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <random>
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
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help,h", "display this help message")
      ("generate,g", po::value<string>(), "create sequence generator")
      ("pipe,p", po::value<vector<string> >(), "pipe (compose) machine(s)")
      ("concat,c", po::value<vector<string> >(), "concatenate machine(s)")
      ("union,u", po::value<string>(), "take union with machine")
      ("weight,w", po::value<string>(), "parameterize union")
      ("reverse,R", "reverse")
      ("revcomp,r", "reverse complement")
      ("kleene,k", "make Kleene closure")
      ("loop,l", po::value<string>(), "parameterize Kleene closure")
      ("accept,a", po::value<string>(), "pipe to sequence acceptor")
      ("flip,f", "flip input/output")
      ("null,n", "pipe to null transducer")
      ("save,s", po::value<string>(), "save machine")
      ("fit,F", "Baum-Welch parameter fit")
      ("params,P", po::value<string>(), "parameter file")
      ("constraints,C", po::value<string>(), "constraints file")
      ("data,D", po::value<string>(), "sequence-pair file")
      ("align,A", "Viterbi sequence alignment")
      ("verbose,v", po::value<int>()->default_value(2), "verbosity level")
      ("log", po::value<vector<string> >(), "log specified function")
      ("nocolor", "log in monochrome")
      ;

    po::positional_options_description p;
    p.add("pipe", -1);

    po::variables_map vm;
    po::store (po::command_line_parser(argc,argv).options(desc).positional(p).run(), vm);
    po::notify(vm);    

    // parse args
    if (vm.count("help")) {
      cout << desc << "\n";
      return 1;
    }

    logger.parseLogArgs (vm);

    // create transducer
    Machine machine;

    // Generator
    if (vm.count("generate")) {
      const NamedInputSeq inSeq = NamedInputSeq::fromFile (vm.at("generate").as<string>());
      LogThisAt(2,"Creating generator for sequence " << inSeq.name << endl);
      machine = Machine::generator (inSeq.name, inSeq.seq);
    }

    // Compositions
    if (vm.count("pipe")) {
      const vector<string> machines = vm.at("pipe").as<vector<string> >();
      for (const auto& filename: machines) {
	LogThisAt(2,"Loading transducer " << filename << endl);
	const Machine loaded = MachineLoader::fromFile(filename);
	machine = machine.nStates() ? Machine::compose (machine, loaded) : loaded;
      }
    }

    // Concatenations
    if (vm.count("concat")) {
      const vector<string> machines = vm.at("concat").as<vector<string> >();
      for (const auto& filename: machines) {
	LogThisAt(2,"Concatenating transducer " << filename << endl);
	const Machine concat = MachineLoader::fromFile(filename);
	machine = machine.nStates() ? Machine::concatenate (machine, concat) : concat;
      }
    }

    // Union
    Require (!vm.count("weight") || vm.count("union"), "Can't specify --weight without --union");
    if (vm.count("union")) {
      const string filename = vm.at("union").as<string>();
      LogThisAt(2,"Taking union with transducer " << filename << endl);
      const Machine uni = MachineLoader::fromFile(filename);
      machine = vm.count("weight")
	? Machine::unionOf (uni, machine, WeightExpr(vm.at("weight").as<string>()))
	: machine = Machine::unionOf (uni, machine);
    }

    // Reverse
    if (vm.count("reverse"))
      machine = machine.reverse();

    // Reverse complement
    if (vm.count("revcomp")) {
      const vguard<OutputSymbol> outAlph = machine.outputAlphabet();
      const set<OutputSymbol> outAlphSet (outAlph.begin(), outAlph.end());
      machine = Machine::compose (machine.reverse(),
				  MachinePresets::makePreset ((outAlphSet.count(string("U")) || outAlphSet.count(string("u")))
							      ? "comprna"
							      : "compdna"));
    }

    // Kleene closure
    if (vm.count("kleene") || vm.count("loop")) {
      LogThisAt(2,"Making Kleene closure" << endl);
      machine = vm.count("loop")
	? machine.kleeneClosure (WeightExpr (vm.at("loop").as<string>()))
	: machine.kleeneClosure();
    }
    
    // Acceptor
    if (vm.count("accept")) {
      const NamedOutputSeq outSeq = NamedInputSeq::fromFile (vm.at("accept").as<string>());
      LogThisAt(2,"Creating acceptor for sequence " << outSeq.name << endl);
      const Machine acceptor = Machine::acceptor (outSeq.name, outSeq.seq);
      machine = machine.nStates() ? Machine::compose(machine,acceptor) : acceptor;
    }

    // Flip
    if (vm.count("flip"))
      machine = machine.flipInOut();
    
    // Null
    if (vm.count("null")) {
      LogThisAt(2,"Creating null transducer" << endl);
      machine = machine.nStates() ? Machine::compose (machine, Machine::null()) : Machine::null();
    }

    // no transducer yet?
    Require (machine.nStates(), "Please specify a transducer (-h for options)");
    
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
