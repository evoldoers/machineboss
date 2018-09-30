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
#include "../src/forward.h"
#include "../src/counts.h"
#include "../src/util.h"
#include "../src/schema.h"
#include "../src/hmmer.h"
#include "../src/csv.h"
#include "../src/compiler.h"

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
      ("generate-chars,g", po::value<string>(), "generator for explicit character sequence '<<'")
      ("generate-one", po::value<string>(), "generator for any one of specified characters")
      ("generate-wild", po::value<string>(), "generator for Kleene closure over specified characters")
      ("generate-fasta", po::value<string>(), "generator for FASTA-format sequence")
      ("generate", po::value<string>(), "sequence generator for JSON-format sequence")
      ("accept-chars,a", po::value<string>(), "acceptor for explicit character sequence '>>'")
      ("accept-one", po::value<string>(), "acceptor for any one of specified characters")
      ("accept-wild", po::value<string>(), "acceptor for Kleene closure over specified characters")
      ("accept-fasta", po::value<string>(), "acceptor for FASTA-format sequence")
      ("accept", po::value<string>(), "sequence acceptor for JSON-format sequence")
      ("echo-one", po::value<string>(), "identity for any one of specified characters")
      ("echo-wild", po::value<string>(), "identity for Kleene closure over specified characters")
      ("weight,w", po::value<string>(), "weighted null transition '#'")
      ("hmmer,H", po::value<string>(), "load machine from HMMER3 model file")
      ("csv,V", po::value<string>(), "load machine from CSV file")
      ;

    po::options_description prefixOpts("Prefix operators");
    prefixOpts.add_options()
      ("repeat", po::value<int>(), "repeat N times")
      ("reverse,e", "reverse")
      ("revcomp,r", "reverse-complement '~'")
      ("transpose,t", "transpose: swap input/output")
      ("sort-sum", "topologically sort, eliminating silent cycles")
      ("sort-break", "topologically sort, breaking silent cycles (faster than --sort-sum, but less precise)")
      ("eliminate,n", "eliminate all silent transitions")
      ;

    po::options_description postfixOpts("Postfix operators");
    postfixOpts.add_options()
      ("zero-or-one,z", "union with null '?'")
      ("kleene-star,k", "Kleene star '*'")
      ("kleene-plus,K", "Kleene plus '+'")
      ("reciprocal", "invert all weight expressions")
      ("weight-input", po::value<string>(), "apply weight parameter with given prefix to inputs")
      ("weight-output", po::value<string>(), "apply weight parameter with given prefix to outputs")
      ;

    po::options_description infixOpts("Infix operators");
    infixOpts.add_options()
      ("compose-sum,m", "compose, summing out silent cycles '=>'")
      ("compose", "compose, breaking silent cycles (faster)")
      ("compose-unsort", "compose, leaving silent cycles")
      ("concatenate,c", "concatenate '.'")
      ("intersect-sum,i", "intersect, summing out silent cycles '&&'")
      ("intersect", "intersect, breaking silent cycles (faster)")
      ("intersect-unsort", "intersect, leaving silent cycles")
      ("union,u", "union '||'")
      ("loop,o", "loop: x '?+' y = x(y.x)*")
      ;

    po::options_description miscOpts("Miscellaneous");
    miscOpts.add_options()
      ("begin,B", "left bracket '('")
      ("end,E", "right bracket ')'")
      ;

    po::options_description appOpts("Transducer application");
    appOpts.add_options()
      ("save,S", po::value<string>(), "save machine to file")
      ("graphviz,G", "write machine in Graphviz DOT format")
      ("memoize,M", "memoize repeated expressions for compactness")
      ("showparams,W", "show unbound parameters in final machine")

      ("params,P", po::value<vector<string> >(), "load parameters (JSON)")
      ("functions,F", po::value<vector<string> >(), "load functions & constants (JSON)")
      ("norms,N", po::value<vector<string> >(), "load normalization constraints (JSON)")
      ("data,D", po::value<vector<string> >(), "load sequence-pairs (JSON)")
      ("input-fasta,I", po::value<string>(), "load input sequence(s) from FASTA file")
      ("input-chars", po::value<string>(), "specify input character sequence explicitly")
      ("output-fasta,O", po::value<string>(), "load output sequence(s) from FASTA file")
      ("output-chars", po::value<string>(), "specify output character sequence explicitly")

      ("train,T", "Baum-Welch parameter fit")
      ("wiggle-room,R", po::value<int>(), "wiggle room (allowed departure from training alignment)")
      ("align,A", "Viterbi sequence alignment")
      ("loglike,L", "Forward log-likelihood calculation")
      ("counts,C", "Forward-Backward counts (derivatives of log-likelihood with respect to logs of parameters)")
      ;

    po::options_description compOpts("Parser-generator");
    compOpts.add_options()
      ("codegen", po::value<string>(), "generate parser code, save to specified filename prefix")
      ("cpp64", "generate C++ dynamic programming code (64-bit)")
      ("cpp32", "generate C++ dynamic programming code (32-bit)")
      ("js", "generate JavaScript dynamic programming code")
      ("showcells", "include debugging output in generated code")
      ("inseq", po::value<string>(), "input sequence type (String, Intvec, Profile)")
      ("outseq", po::value<string>(), "output sequence type (String, Intvec, Profile)")
      ;

    po::options_description transOpts("");
    transOpts.add(createOpts).add(prefixOpts).add(postfixOpts).add(infixOpts).add(miscOpts);

    po::options_description helpOpts("");
    helpOpts.add(generalOpts).add(createOpts).add(prefixOpts).add(postfixOpts).add(infixOpts).add(miscOpts).add(appOpts).add(compOpts);

    po::options_description parseOpts("");
    parseOpts.add(generalOpts).add(appOpts).add(compOpts);

    map<string,string> alias;
    alias[string("<<")] = "--generate-chars";
    alias[string(">>")] = "--accept-chars";
    alias[string("=>")] = "--compose-sum";
    alias[string(".")] = "--concatenate";
    alias[string("&&")] = "--intersect-sum";
    alias[string("||")] = "--union";
    alias[string("?")] = "--zero-or-one";
    alias[string("*")] = "--kleene-star";
    alias[string("+")] = "--kleene-plus";
    alias[string("?+")] = "--loop";
    alias[string("#")] = "--weight";
    alias[string("~")] = "--revcomp";
    alias[string("(")] = "--begin";
    alias[string(")")] = "--end";

    alias[string("--recip")] = "--reciprocal";
    alias[string("--concat")] = "--concatenate";
    alias[string("--or")] = "--union";

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
	  machine = Machine::compose (machines.back(), machine, true, true, Machine::SumSilentCycles);
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
	  const NamedInputSeq inSeq = JsonLoader<NamedInputSeq>::fromFile (getArg());
	  m = Machine::generator (inSeq.seq, inSeq.name);
	} else if (command == "--generate-fasta") {
	  const vguard<FastSeq> inSeqs = readFastSeqs (getArg().c_str());
	  Require (inSeqs.size() == 1, "--generate-fasta file must contain exactly one FASTA-format sequence");
	  m = Machine::generator (splitToChars (inSeqs[0].seq), inSeqs[0].name);
	} else if (command == "--generate-chars") {
	  const string seq = getArg();
	  m = Machine::generator (splitToChars (seq), seq);
	} else if (command == "--generate-wild") {
	  const string chars = getArg();
	  m = Machine::wildGenerator (splitToChars (chars));
	} else if (command == "--generate-one") {
	  const string chars = getArg();
	  m = Machine::wildSingleGenerator (splitToChars (chars));
	} else if (command == "--accept") {
	  const NamedOutputSeq outSeq = JsonLoader<NamedOutputSeq>::fromFile (getArg());
	  m = Machine::acceptor (outSeq.seq, outSeq.name);
	} else if (command == "--accept-fasta") {
	  const vguard<FastSeq> outSeqs = readFastSeqs (getArg().c_str());
	  Require (outSeqs.size() == 1, "--accept-fasta file must contain exactly one FASTA-format sequence");
	  m = Machine::acceptor (splitToChars (outSeqs[0].seq), outSeqs[0].name);
	} else if (command == "--accept-chars") {
	  const string seq = getArg();
	  m = Machine::acceptor (splitToChars (seq), seq);
	} else if (command == "--accept-wild") {
	  const string chars = getArg();
	  m = Machine::wildAcceptor (splitToChars (chars));
	} else if (command == "--accept-one") {
	  const string chars = getArg();
	  m = Machine::wildSingleAcceptor (splitToChars (chars));
	} else if (command == "--echo-wild") {
	  const string chars = getArg();
	  m = Machine::wildEcho (splitToChars (chars));
	} else if (command == "--echo-one") {
	  const string chars = getArg();
	  m = Machine::wildSingleEcho (splitToChars (chars));
	} else if (command == "--sort-sum")
	  m = nextMachine().advanceSort().advancingMachine();
	else if (command == "--sort-break")
	  m = nextMachine().advanceSort().dropSilentBackTransitions();
	else if (command == "--compose-sum")
	  m = Machine::compose (popMachine(), nextMachine(), true, true, Machine::SumSilentCycles);
	else if (command == "--compose")
          m = Machine::compose (popMachine(), nextMachine(), true, true, Machine::BreakSilentCycles);
	else if (command == "--compose-unsort")
	  m = Machine::compose (popMachine(), nextMachine(), true, true, Machine::LeaveSilentCycles);
	else if (command == "--concatenate")
	  m = Machine::concatenate (popMachine(), nextMachine());
	else if (command == "--intersect-sum")
	  m = Machine::intersect (popMachine(), nextMachine(), Machine::SumSilentCycles);
	else if (command == "--intersect")
	  m = Machine::intersect (popMachine(), nextMachine(), Machine::BreakSilentCycles);
	else if (command == "--intersect-unsort")
	  m = Machine::intersect (popMachine(), nextMachine(), Machine::LeaveSilentCycles);
	else if (command == "--union")
	  m = Machine::takeUnion (popMachine(), nextMachine());
	else if (command == "--zero-or-one")
	  m = Machine::zeroOrOne (popMachine());
	else if (command == "--kleene-star")
	  m = Machine::kleeneStar (popMachine());
	else if (command == "--kleene-plus")
	  m = Machine::kleenePlus (popMachine());
	else if (command == "--repeat") {
	  const int nReps = stoi (getArg());
	  Require (nReps > 0, "--repeat requires minimum one repetition");
	  const Machine unit = nextMachine();
	  m = unit;
	  for (int n = 1; n < nReps; ++n)
	    m = Machine::concatenate (m, unit);
	} else if (command == "--loop")
	  m = Machine::kleeneLoop (popMachine(), nextMachine());
	else if (command == "--eliminate")
	  m = nextMachine().eliminateSilentTransitions();
	else if (command == "--reverse")
	  m = nextMachine().reverse();
	else if (command == "--revcomp") {
	  const Machine r = nextMachine();
	  const vguard<OutputSymbol> outAlph = m.outputAlphabet();
	  const set<OutputSymbol> outAlphSet (outAlph.begin(), outAlph.end());
	  m = Machine::compose (r.reverse(),
				MachinePresets::makePreset ((outAlphSet.count(string("U")) || outAlphSet.count(string("u")))
							    ? "comprna"
							    : "compdna"),
				true, true, Machine::SumSilentCycles);
	} else if (command == "--transpose")
	  m = nextMachine().transpose();
	else if (command == "--weight") {
	  const string wArg = getArg();
	  json wj;
	  WeightExpr w;
	  try {
	    wj = json::parse(wArg);
	    if (MachineSchema::validate ("expr", wj))
	      w = WeightAlgebra::fromJson (wj);
	  } catch (...) {
	    const char* wc = wArg.c_str();
	    char* p;
	    const long intValue = strtol (wc, &p, 10);
	    if (*p) {
	      // integer conversion failed
	      const double doubleValue = strtod (wc, &p);
	      if (*p) {
		// double conversion failed
		w = WeightAlgebra::param (wArg);
	      } else
		w = WeightAlgebra::doubleConstant (doubleValue);
	    }
	    else
	      w = WeightAlgebra::intConstant (intValue);
	  }
	  m = Machine::singleTransition (w);
	} else if (command == "--weight-input") {
	  m = popMachine().weightInputs (getArg().c_str());
	} else if (command == "--weight-output") {
	  m = popMachine().weightOutputs (getArg().c_str());
	} else if (command == "--reciprocal") {
	  m = popMachine().reciprocal();
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
	else if (command == "--hmmer") {
	  HmmerModel hmmer;
	  ifstream infile (getArg());
	  Require (infile, "HMMer model file not found");
	  hmmer.read (infile);
	  m = hmmer.machine();
	} else if (command == "--csv") {
	  CSVProfile csv;
	  ifstream infile (getArg());
	  Require (infile, "CSV file not found");
	  csv.read (infile);
	  m = csv.machine();
	} else {
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
    Machine machine = reduceMachines();

    // load parameters and constraints
    ParamAssign seed;
    if (vm.count("params"))
      JsonLoader<ParamAssign>::readFiles (seed, vm.at("params").as<vector<string> >());

    ParamFuncs funcs;
    if (vm.count("functions"))
      JsonLoader<ParamFuncs>::readFiles (funcs, vm.at("functions").as<vector<string> >());

    Constraints constraints;
    if (vm.count("norms"))
      JsonLoader<Constraints>::readFiles (constraints, vm.at("norms").as<vector<string> >());

    // if constraints or parameters were specified without a training or alignment step,
    // then add them to the model now; otherwise, save them for later
    if ((vm.count("params") || vm.count("functions") || vm.count("norms"))
	&& !(vm.count("train") || vm.count("loglike") || vm.count("align"))) {
      machine.defs = seed;
      machine.defs.defs.insert (funcs.defs.begin(), funcs.defs.end());
      machine.cons = constraints;
    }
    
    // output transducer
    function<void(ostream&)> showMachine = [&](ostream& out) {
      if (vm.count("graphviz"))
	machine.writeDot (out);
      else
	machine.writeJson (out, vm.count("memoize"), vm.count("showparams"));
    };
    if (vm.count("save")) {
      const string savefile = vm.at("save").as<string>();
      ofstream out (savefile);
      showMachine (out);
    } else if (!vm.count("train") && !vm.count("loglike") && !vm.count("align") && !vm.count("counts") && !vm.count("codegen"))
      showMachine (cout);

    // code generation
    function<Compiler::SeqType(const char*,const vguard<string>&)> getSeqType = [&](const char* tag, const vguard<string>& alph) {
      if (!vm.count(tag))
	return Compiler::isCharAlphabet(alph) ? Compiler::SeqType::String : Compiler::SeqType::IntVec;
      const char c = tolower (vm.at(tag).as<string>()[0]);
      if (c != 's' && c != 'i' && c != 'p')
	Fail ("Sequence type must be S (string), I (integer vector) or P (profile weight matrix)");
      return c == 's' ? Compiler::SeqType::String : (c == 'i' ? Compiler::SeqType::IntVec : Compiler::SeqType::Profile);
    };
    function<void(Compiler&)> compileMachine = [&](Compiler& compiler) {
      const Compiler::SeqType xSeqType = getSeqType ("inseq", machine.inputAlphabet());
      const Compiler::SeqType ySeqType = getSeqType ("outseq", machine.outputAlphabet());
      const string filenamePrefix = vm.at("codegen").as<string>();
      compiler.showCells = vm.count("showcells");
      compiler.compileForward (machine, xSeqType, ySeqType, filenamePrefix.c_str());
    };
    Assert (vm.count("cpp32") + vm.count("cpp64") + vm.count("js") < 2, "Options --cpp32, --cpp64 and --js are mutually incompatible; choose a target language");
    if (vm.count("codegen")) {
      if (vm.count("js")) {
	JavaScriptCompiler compiler;
	compileMachine (compiler);
      } else {
	CPlusPlusCompiler compiler (vm.count("cpp64"));
	compileMachine (compiler);
      }
    }

    // load data
    SeqPairList data;
    // list of I/O pairs specified?
    if (vm.count("data"))
      JsonLoader<SeqPairList>::readFiles (data, vm.at("data").as<vector<string> >());

    // individual inputs or outputs specified?
    vguard<FastSeq> inSeqs, outSeqs;
    if (vm.count("input-fasta"))
      readFastSeqs (vm.at("input-fasta").as<string>().c_str(), inSeqs);
    if (vm.count("output-fasta"))
      readFastSeqs (vm.at("output-fasta").as<string>().c_str(), outSeqs);
    if (vm.count("input-chars")) {
      const string seq = vm.at("input-chars").as<string>();
      inSeqs.push_back (FastSeq::fromSeq (seq, seq));
    }
    if (vm.count("output-chars")) {
      const string seq = vm.at("output-chars").as<string>();
      outSeqs.push_back (FastSeq::fromSeq (seq, seq));
    }

    // if inputs/outputs specified individually, create all input-output pairs
    if (inSeqs.empty() && !outSeqs.empty() && machine.inputAlphabet().empty())
      inSeqs.push_back (FastSeq());
    else if (outSeqs.empty() && !inSeqs.empty() && machine.outputAlphabet().empty())
      outSeqs.push_back (FastSeq());
    for (const auto& inSeq: inSeqs)
      for (const auto& outSeq: outSeqs)
	data.seqPairs.push_back (SeqPair ({ NamedInputSeq ({ inSeq.name, splitToChars (inSeq.seq) }), NamedOutputSeq ({ outSeq.name, splitToChars (outSeq.seq) }) }));

    // after all that, do we have data? did we need data?
    const bool needData = vm.count("train") || vm.count("loglike") || vm.count("align") || vm.count("counts");
    const bool noIO = machine.inputAlphabet().empty() && machine.outputAlphabet().empty();
    if (needData && data.seqPairs.empty() && noIO)
      data.seqPairs.push_back (SeqPair());  // if the model has no I/O, then add an automatic pair of empty, nameless sequences (the only possible evidence)
    const bool gotData = !data.seqPairs.empty();
    Require (!gotData || needData, "No point in specifying input/output data without --train, --loglike, --align, or --counts");

    // fit parameters
    Params params;
    if (vm.count("train")) {
      Require ((vm.count("norms") || !machine.cons.empty())
	       && (gotData || noIO),
	       "To fit parameters, please specify a constraints file and (for machines with input/output) a data file");
      MachineFitter fitter;
      fitter.machine = machine;
      if (vm.count("norms"))
	fitter.constraints = constraints;
      fitter.constants = funcs;
      fitter.seed = fitter.allConstraints().defaultParams().combine (seed);
      params = vm.count("wiggle-room") ? fitter.fit(data,vm.at("wiggle-room").as<int>()) : fitter.fit(data);
      cout << JsonLoader<Params>::toJsonString(params) << endl;
    } else
      params = funcs.combine (seed);

    // compute sequence log-likelihoods
    if (vm.count("loglike")) {
      const EvaluatedMachine eval (machine, params);
      cout << "[";
      size_t n = 0;
      for (const auto& seqPair: data.seqPairs) {
	const ForwardMatrix forward (eval, seqPair);
	cout << (n++ ? ",\n " : "")
	     << "[\"" << escaped_str(seqPair.input.name)
	     << "\",\"" << escaped_str(seqPair.output.name)
	     << "\"," << forward.logLike() << "]";
      }
      cout << "]\n";
    }

    // compute counts
    if (vm.count("counts")) {
      const EvaluatedMachine eval (machine, params);
      const MachineCounts counts (eval, data);
      counts.writeParamCountsJson (cout, machine, params);
      cout << endl;
    }

    // align sequences
    if (vm.count("align")) {
      Require (gotData, "To align sequences, please specify a data file");
      const EvaluatedMachine eval (machine, params);
      SeqPairList alignResults;
      for (const auto& seqPair: data.seqPairs) {
	const ViterbiMatrix viterbi (eval, seqPair);
	const MachineBoundPath path (viterbi.path (machine), machine);
	alignResults.seqPairs.push_back (SeqPair::seqPairFromPath (path, seqPair.input.name.c_str(), seqPair.output.name.c_str()));
      }
      alignResults.writeJson (cout);
    }
    
  } catch (const std::exception& e) {
    cerr << e.what() << endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}
