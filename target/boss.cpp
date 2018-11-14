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
#include "../src/ctc.h"
#include "../src/beam.h"
#include "../src/net.h"

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
      ("generate-iid", po::value<string>(), "as --generate-wild, but followed by --weight-output " MachineParamPrefix)
      ("generate-uniform", po::value<string>(), "as --generate-iid, but weights outputs by 1/(output alphabet size)")
      ("generate-fasta", po::value<string>(), "generator for FASTA-format sequence")
      ("generate-csv", po::value<string>(), "create generator from CSV file")
      ("generate-json", po::value<string>(), "sequence generator for JSON-format sequence")
      ("generate-uniprot", po::value<string>(), "create generator from UniProt ID (e.g. P12345)")
      ("accept-chars,a", po::value<string>(), "acceptor for explicit character sequence '>>'")
      ("accept-one", po::value<string>(), "acceptor for any one of specified characters")
      ("accept-wild", po::value<string>(), "acceptor for Kleene closure over specified characters")
      ("accept-iid", po::value<string>(), "as --accept-wild, but followed by --weight-input " MachineParamPrefix)
      ("accept-uniform", po::value<string>(), "as --accept-iid, but weights outputs by 1/(input alphabet size)")
      ("accept-fasta", po::value<string>(), "acceptor for FASTA-format sequence")
      ("accept-csv", po::value<string>(), "create acceptor from CSV file")
      ("accept-merge-csv", po::value<string>(), "create acceptor from CSV file, merging consecutively repeated characters as in Graves (2006) 'Connectionist Temporal Classification'")
      ("accept-json", po::value<string>(), "sequence acceptor for JSON-format sequence")
      ("echo-one", po::value<string>(), "identity for any one of specified characters")
      ("echo-wild", po::value<string>(), "identity for Kleene closure over specified characters")
      ("echo-chars", po::value<string>(), "identity for explicit character sequence")
      ("echo-fasta", po::value<string>(), "identity for FASTA-format sequence")
      ("echo-json", po::value<string>(), "identity for JSON-format sequence")
      ("weight,w", po::value<string>(), "weighted null transition '#'")
      ("hmmer,H", po::value<string>(), "create generator from HMMER3 model file")
      ("pfam", po::value<string>(), "create generator from PFAM ID (e.g. Piwi)")
      ("dfam", po::value<string>(), "create generator from DFAM ID (e.g. DF0004136)")
      ;

    po::options_description postfixOpts("Postfix operators");
    postfixOpts.add_options()
      ("zero-or-one,z", "union with null '?'")
      ("kleene-star,k", "Kleene star '*'")
      ("kleene-plus,K", "Kleene plus '+'")
      ("count-copies", po::value<string>(), "Kleene star with dummy counting parameter")
      ("repeat", po::value<int>(), "repeat N times")
      ("reverse,e", "reverse")
      ("revcomp,r", "reverse-complement '~'")
      ("local-input", "add flanking delete states: partially match input")
      ("local-output", "add flanking insert states: partially match ouput")
      ("local-either", "add flanking insert or delete states: partially match either input or output at each end")
      ("local-both", "add flanking insert & delete states: partially match input and/or output")
      ("double-strand", "union of machine with its reverse complement")
      ("transpose,t", "transpose: swap input/output")
      ("downsample", po::value<double>(), "keep only specified proportion of transitions, discarding those with lowest posterior weight")
      ("downsample-prob", po::value<double>(), "keep only transitions above this posterior weight threshold, along with any required for ergodicity")
      ("joint-norm", "normalize jointly (outgoing transition weights sum to 1)")
      ("cond-norm", "normalize conditionally (outgoing transition weights for each input symbol sum to 1)")
      ("sort", "topologically sort, eliminating silent cycles")
      ("sort-fast", "topologically sort, breaking silent cycles (faster than --sort, but destructive)")
      ("sort-cyclic", "topologically sort if possible, but preserve silent cycles")
      ("decode-sort", "topologically sort non-outputting transition graph")
      ("encode-sort", "topologically sort non-inputting transition graph")
      ("full-sort", "topologically sort entire transition graph, not just silent transitions")
      ("eliminate,n", "eliminate all silent transitions")
      ("eliminate-states", "eliminate all states whose only outgoing transition is silent")
      ("pad", "pad with \"dummy\" start & end states")
      ("reciprocal", "element-wise reciprocal: invert all weight expressions")
      ("weight-input", po::value<string>(), "multiply input weights by parameter with given prefix")
      ("weight-output", po::value<string>(), "multiply output weights by parameter with given prefix")
      ("param-odds-ratio", po::value<string>(), "divide output weights by parameter with given prefix")
      ("uniform-odds-ratio", "divide output weights by uniform distribution over output alphabet")
      ;

    po::options_description infixOpts("Infix operators");
    infixOpts.add_options()
      ("compose,m", "compose, summing out silent cycles '=>'")
      ("compose-fast", "compose, breaking silent cycles (faster, destructive)")
      ("compose-cyclic", "compose, leaving silent cycles")
      ("concatenate,c", "concatenate '.'")
      ("intersect,i", "intersect, summing out silent cycles '&&'")
      ("intersect-fast", "intersect, breaking silent cycles (faster, destructive)")
      ("intersect-cyclic", "intersect, leaving silent cycles")
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
      ("evaluate", "evaluate all transition weights in final machine")
      ("define-exprs", "define and re-use repeated (sub)expressions, for compactness")
      ("show-params", "show unbound parameters in final machine")
      ("use-defaults,U", "use defaults (uniform distributions, unit rates) for unbound parameters; this option is implicit when training")
      ("name-states", "use state id, rather than number, to identify transition destinations")

      ("params,P", po::value<vector<string> >(), "load parameters (JSON)")
      ("functions,F", po::value<vector<string> >(), "load functions & constants (JSON)")
      ("constraints,N", po::value<vector<string> >(), "load normalization constraints (JSON)")
      ("data,D", po::value<vector<string> >(), "load sequence-pairs (JSON)")
      ("input-fasta,I", po::value<string>(), "load input sequence(s) from FASTA file")
      ("input-json", po::value<string>(), "load input sequence from JSON file")
      ("input-chars", po::value<string>(), "specify input character sequence explicitly")
      ("input-uniprot", po::value<string>(), "specify input sequence by UniProt ID (e.g. P12345)")
      ("output-fasta,O", po::value<string>(), "load output sequence(s) from FASTA file")
      ("output-json", po::value<string>(), "load output sequence from JSON file")
      ("output-chars", po::value<string>(), "specify output character sequence explicitly")
      ("output-uniprot", po::value<string>(), "specify output sequence by UniProt ID (e.g. P12345)")

      ("train,T", "Baum-Welch parameter fit")
      ("wiggle-room,R", po::value<int>(), "wiggle room (allowed departure from training alignment)")
      ("align,A", "Viterbi sequence alignment")
      ("loglike,L", "Forward log-likelihood calculation")
      ("counts,C", "Forward-Backward counts (derivatives of log-likelihood with respect to logs of parameters)")
      ("beam-decode,Z", "find most likely input by beam search")
      ("beam-width", po::value<size_t>(), (string("number of sequences to track during beam search (default ") + to_string((size_t)DefaultBeamWidth) + ")").c_str())
      ("prefix-decode", "find most likely input by CTC prefix search")
      ("prefix-backtrack", po::value<long>(), "specify max backtracking length for CTC prefix search")
      ("cool-decode", "find most likely input by simulated annealing")
      ("mcmc-decode", "find most likely input by MCMC search")
      ("decode-steps", po::value<int>(), "simulated annealing steps per initial symbol")
      ("beam-encode,Y", "find most likely output by beam search")
      ("prefix-encode", "find most likely output by CTC prefix search")
      ("random-encode", "sample random output by stochastic prefix search")
      ("seed", po::value<int>(), "random number seed")
      ;

    po::options_description compOpts("Parser-generator");
    compOpts.add_options()
      ("codegen", po::value<string>(), "generate parser code, save to specified directory")
      ("cpp64", "generate C++ dynamic programming code (64-bit)")
      ("cpp32", "generate C++ dynamic programming code (32-bit)")
      ("js", "generate JavaScript dynamic programming code")
      ("showcells", "include debugging output in generated code")
      ("inseq", po::value<string>(), "input sequence type (String, Intvec, Profile)")
      ("outseq", po::value<string>(), "output sequence type (String, Intvec, Profile)")
      ;

    po::options_description transOpts("");
    transOpts.add(createOpts).add(postfixOpts).add(infixOpts).add(miscOpts);

    po::options_description helpOpts("");
    helpOpts.add(generalOpts).add(createOpts).add(postfixOpts).add(infixOpts).add(miscOpts).add(appOpts).add(compOpts);

    po::options_description parseOpts("");
    parseOpts.add(generalOpts).add(appOpts).add(compOpts);

    map<string,string> alias;
    alias[string("<<")] = "--generate-chars";
    alias[string(">>")] = "--accept-chars";
    alias[string("=>")] = "--compose";
    alias[string(".")] = "--concatenate";
    alias[string("&&")] = "--intersect";
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

    const regex presetAlphRegex ("^--(generate|accept|echo)-(one|wild|iid|uniform)-(dna|rna|aa)$");
    map<string,string> presetAlph;
    presetAlph[string("dna")] = "ACGT";
    presetAlph[string("rna")] = "ACGU";
    presetAlph[string("aa")] = "ACDEFGHIKLMNPQRSTVWY";

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
	// reduce immediately (else other operators have higher precedence than implicit reduction)
	if (machines.size() > 1)
	  machines.push_back (reduceMachines());
      };
      nextMachineForCommand = [&] (const string& lastCommand) -> Machine {
	if (args.empty()) {
	  cout << helpOpts << endl;
	  throw runtime_error (lastCommand.size() ? (string("Missing argument for ") + lastCommand) : string("Missing command"));
	}
	string arg = args.front();
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
	auto revCompMachine = [&] (const Machine& r) -> Machine {
	  const vguard<OutputSymbol> outAlph = r.outputAlphabet();
	  const set<OutputSymbol> outAlphSet (outAlph.begin(), outAlph.end());
	  return Machine::compose (r.reverse(),
				   MachinePresets::makePreset ((outAlphSet.count(string("U")) || outAlphSet.count(string("u")))
							       ? "comprna"
							       : "compdna"),
				   true, true, Machine::SumSilentCycles);
	};

	smatch presetAlphMatch;
	if (regex_search (arg, presetAlphMatch, presetAlphRegex)) {
	  args.push_front (presetAlph[presetAlphMatch.str(3)]);
	  arg = string("--") + presetAlphMatch.str(1) + "-" + presetAlphMatch.str(2);
	}

	if (alias.count (arg))
	  arg = alias.at (arg);

	const po::option_description* desc = NULL;
	if (arg[0] == '-') {
	  desc = transOpts.find_nothrow (arg, false);
	  if (!desc && arg.size() > 1 && arg[1] == '-')
	    desc = transOpts.find_nothrow (arg.substr(2), false);
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
	else if (command == "--generate-json") {
	  const NamedInputSeq inSeq = JsonLoader<NamedInputSeq>::fromFile (getArg());
	  m = Machine::generator (inSeq.seq, inSeq.name);
	} else if (command == "--generate-fasta") {
	  const vguard<FastSeq> inSeqs = readFastSeqs (getArg().c_str());
	  Require (inSeqs.size() == 1, "--generate-fasta file must contain exactly one FASTA-format sequence");
	  m = Machine::generator (splitToChars (inSeqs[0].seq), inSeqs[0].name);
	} else if (command == "--generate-uniprot") {
	  const FastSeq fs = getUniprot (getArg());
	  m = Machine::generator (splitToChars (fs.seq), fs.name);
	} else if (command == "--generate-chars") {
	  const string seq = getArg();
	  m = Machine::generator (splitToChars (seq), seq);
	} else if (command == "--generate-wild") {
	  const string chars = getArg();
	  m = Machine::wildGenerator (splitToChars (chars));
	} else if (command == "--generate-iid") {
	  const string chars = getArg();
	  m = Machine::wildGenerator (splitToChars (chars)).weightOutputs (MachineParamPrefix);
	} else if (command == "--generate-uniform") {
	  const string chars = getArg();
	  m = Machine::wildGenerator (splitToChars (chars)).weightOutputsUniformly();
	} else if (command == "--generate-one") {
	  const string chars = getArg();
	  m = Machine::wildSingleGenerator (splitToChars (chars));
	} else if (command == "--accept-json") {
	  const NamedOutputSeq outSeq = JsonLoader<NamedOutputSeq>::fromFile (getArg());
	  m = Machine::acceptor (outSeq.seq, outSeq.name);
	} else if (command == "--accept-fasta") {
	  const vguard<FastSeq> outSeqs = readFastSeqs (getArg().c_str());
	  Require (outSeqs.size() == 1, "--accept-fasta file must contain exactly one FASTA-format sequence");
	  m = Machine::acceptor (splitToChars (outSeqs[0].seq), outSeqs[0].name);
	} else if (command == "--accept-uniprot") {
	  const FastSeq fs = getUniprot (getArg());
	  m = Machine::acceptor (splitToChars (fs.seq), fs.name);
	} else if (command == "--accept-chars") {
	  const string seq = getArg();
	  m = Machine::acceptor (splitToChars (seq), seq);
	} else if (command == "--accept-wild") {
	  const string chars = getArg();
	  m = Machine::wildAcceptor (splitToChars (chars));
	} else if (command == "--accept-iid") {
	  const string chars = getArg();
	  m = Machine::wildAcceptor (splitToChars (chars)).weightInputs (MachineParamPrefix);
	} else if (command == "--accept-uniform") {
	  const string chars = getArg();
	  m = Machine::wildAcceptor (splitToChars (chars)).weightInputsUniformly();
	} else if (command == "--accept-one") {
	  const string chars = getArg();
	  m = Machine::wildSingleAcceptor (splitToChars (chars));
	} else if (command == "--echo-wild") {
	  const string chars = getArg();
	  m = Machine::wildEcho (splitToChars (chars));
	} else if (command == "--echo-one") {
	  const string chars = getArg();
	  m = Machine::wildSingleEcho (splitToChars (chars));
	} else if (command == "--echo-chars") {
	  const string seq = getArg();
	  m = Machine::echo (splitToChars (seq), seq);
	} else if (command == "--echo-fasta") {
	  const vguard<FastSeq> inSeqs = readFastSeqs (getArg().c_str());
	  Require (inSeqs.size() == 1, "--echo-fasta file must contain exactly one FASTA-format sequence");
	  m = Machine::echo (splitToChars (inSeqs[0].seq), inSeqs[0].name);
	} else if (command == "--echo-json") {
	  const NamedInputSeq inSeq = JsonLoader<NamedInputSeq>::fromFile (getArg());
	  m = Machine::echo (inSeq.seq, inSeq.name);
	} else if (command == "--sort")
	  m = popMachine().advanceSort().advancingMachine();
	else if (command == "--sort-fast")
	  m = popMachine().advanceSort().dropSilentBackTransitions();
	else if (command == "--sort-cyclic")
	  m = popMachine().advanceSort();
	else if (command == "--joint-norm")
	  m = popMachine().normalizeJointly();
	else if (command == "--cond-norm")
	  m = popMachine().normalizeConditionally();
	else if (command == "--decode-sort")
	  m = popMachine().decodeSort();
	else if (command == "--encode-sort")
	  m = popMachine().encodeSort();
	else if (command == "--full-sort")
	  m = popMachine().toposort();
	else if (command == "--compose")
	  m = Machine::compose (popMachine(), nextMachine(), true, true, Machine::SumSilentCycles);
	else if (command == "--compose-fast")
          m = Machine::compose (popMachine(), nextMachine(), true, true, Machine::BreakSilentCycles);
	else if (command == "--compose-cyclic")
	  m = Machine::compose (popMachine(), nextMachine(), true, true, Machine::LeaveSilentCycles);
	else if (command == "--concatenate")
	  m = Machine::concatenate (popMachine(), nextMachine());
	else if (command == "--intersect")
	  m = Machine::intersect (popMachine(), nextMachine(), Machine::SumSilentCycles);
	else if (command == "--intersect-fast")
	  m = Machine::intersect (popMachine(), nextMachine(), Machine::BreakSilentCycles);
	else if (command == "--intersect-cyclic")
	  m = Machine::intersect (popMachine(), nextMachine(), Machine::LeaveSilentCycles);
	else if (command == "--union")
	  m = Machine::takeUnion (popMachine(), nextMachine());
	else if (command == "--zero-or-one")
	  m = Machine::zeroOrOne (popMachine()).advanceSort();
	else if (command == "--kleene-star")
	  m = Machine::kleeneStar (popMachine()).advanceSort();
	else if (command == "--kleene-plus")
	  m = Machine::kleenePlus (popMachine()).advanceSort();
	else if (command == "--count-copies")
	  m = Machine::kleeneCount (popMachine(), getArg()).advanceSort();
	else if (command == "--repeat") {
	  const int nReps = stoi (getArg());
	  Require (nReps > 0, "--repeat requires minimum one repetition");
	  const Machine unit = popMachine();
	  m = unit;
	  for (int n = 1; n < nReps; ++n)
	    m = Machine::concatenate (m, unit);
	} else if (command == "--loop")
	  m = Machine::kleeneLoop (popMachine(), nextMachine()).advanceSort();
	else if (command == "--eliminate")
	  m = popMachine().eliminateSilentTransitions();
	else if (command == "--eliminate-states")
	  m = popMachine().eliminateRedundantStates();
	else if (command == "--pad")
	  m = popMachine().padWithNullStates();
	else if (command == "--reverse")
	  m = popMachine().reverse();
	else if (command == "--revcomp")
	  m = revCompMachine (popMachine());
	else if (command == "--double-strand") {
	  const WeightExpr half = WeightAlgebra::reciprocal (WeightAlgebra::intConstant (2));
	  const Machine r = popMachine();
	  m = Machine::takeUnion (r, revCompMachine(r), half, half);
	} else if (command == "--transpose")
	  m = popMachine().transpose();
	else if (command == "--downsample")
	  m = popMachine().downsample (stod (getArg()));
	else if (command == "--downsample-prob")
	  m = popMachine().downsample (1., stod (getArg()));
	else if (command == "--local-input") {
	  const Machine core = popMachine();
	  const Machine flank = Machine::wildAcceptor (core.inputAlphabet());
	  return Machine::concatenate (flank, Machine::concatenate (core, flank));
	} else if (command == "--local-output") {
	  const Machine core = popMachine();
	  const Machine flank = Machine::wildGenerator (core.outputAlphabet());
	  return Machine::concatenate (flank, Machine::concatenate (core, flank));
	} else if (command == "--local-either") {
	  const Machine core = popMachine();
	  const Machine flank = Machine::takeUnion (Machine::wildAcceptor (core.inputAlphabet()), Machine::wildGenerator (core.outputAlphabet()));
	  return Machine::concatenate (flank, Machine::concatenate (core, flank));
	} else if (command == "--local-both") {
	  const Machine core = popMachine();
	  const Machine flank = Machine::concatenate (Machine::wildAcceptor (core.inputAlphabet()), Machine::wildGenerator (core.outputAlphabet()));
	  return Machine::concatenate (flank, Machine::concatenate (core, flank));
	} else if (command == "--weight") {
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
	  m = popMachine().pointwiseReciprocal();
	} else if (command == "--param-odds-ratio") {
	  m = popMachine().weightOutputs (getArg().c_str(), true);
	} else if (command == "--uniform-odds-ratio") {
	  m = popMachine().weightOutputsUniformly (true);
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
	} else if (command == "--pfam")
	  m = getPfam(getArg()).machine();
	else if (command == "--dfam")
	  m = getDfam(getArg()).machine();
	else if (command == "--generate-csv") {
	  CSVProfile csv;
	  ifstream infile (getArg());
	  Require (infile, "CSV file not found");
	  csv.read (infile);
	  m = csv.machine();
	} else if (command == "--accept-csv") {
	  CSVProfile csv;
	  ifstream infile (getArg());
	  Require (infile, "CSV file not found");
	  csv.read (infile);
	  m = csv.machine().transpose();
	} else if (command == "--accept-merge-csv") {
	  CSVProfile csv;
	  ifstream infile (getArg());
	  Require (infile, "CSV file not found");
	  csv.read (infile);
	  m = csv.mergingMachine().transpose();
	} else {
	  cout << helpOpts << endl;
	  throw runtime_error (string ("Unknown option: ") + arg);
	}
	LogThisAt(9,"After " << command << ":" << endl << JsonWriter<Machine>::toJsonString(m) << endl);
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
    if (vm.count("constraints"))
      JsonLoader<Constraints>::readFiles (constraints, vm.at("constraints").as<vector<string> >());

    // if constraints or parameters were specified without a training or alignment step,
    // then add them to the model now; otherwise, save them for later
    const bool paramsSpecified = vm.count("params") || vm.count("functions") || vm.count("norms");
    const bool inferenceRequested = vm.count("train") || vm.count("loglike") || vm.count("align") || vm.count("counts") || vm.count("prefix-encode") || vm.count("beam-encode") || vm.count("random-encode") || vm.count("prefix-decode") || vm.count("cool-decode") || vm.count("mcmc-decode") || vm.count("beam-decode");
    const bool evalRequested = vm.count("evaluate");
    if (paramsSpecified	&& (evalRequested || !inferenceRequested)) {
      machine.funcs = machine.funcs.combine(funcs,true).combine(seed,true);
      machine.cons = machine.cons.combine (constraints);
    }

    // evaluate transition weights, if requested
    if (evalRequested) {
      const EvaluatedMachine eval (machine, machine.getParamDefs (vm.count("use-defaults")));
      machine = eval.explicitMachine();
      funcs.clear();
      seed.clear();
      constraints.clear();
    }
    
    // output transducer
    function<void(ostream&)> showMachine = [&](ostream& out) {
      if (vm.count("graphviz"))
	machine.writeDot (out);
      else
	machine.writeJson (out, vm.count("define-exprs"), vm.count("show-params"), vm.count("name-states"));
    };
    if (vm.count("save")) {
      const string savefile = vm.at("save").as<string>();
      ofstream out (savefile);
      showMachine (out);
    } else if (!inferenceRequested && !vm.count("codegen"))
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
    vguard<FastSeq> inFastSeqs, outFastSeqs;
    if (vm.count("input-fasta"))
      readFastSeqs (vm.at("input-fasta").as<string>().c_str(), inFastSeqs);
    if (vm.count("output-fasta"))
      readFastSeqs (vm.at("output-fasta").as<string>().c_str(), outFastSeqs);
    if (vm.count("input-chars")) {
      const string seq = vm.at("input-chars").as<string>();
      inFastSeqs.push_back (FastSeq::fromSeq (seq, seq));
    }
    if (vm.count("output-chars")) {
      const string seq = vm.at("output-chars").as<string>();
      outFastSeqs.push_back (FastSeq::fromSeq (seq, seq));
    }
    if (vm.count ("input-uniprot"))
      inFastSeqs.push_back (getUniprot (vm.at("input-uniprot").as<string>()));
    if (vm.count ("output-uniprot"))
      outFastSeqs.push_back (getUniprot (vm.at("output-uniprot").as<string>()));

    vguard<NamedInputSeq> inSeqs;
    vguard<NamedOutputSeq> outSeqs;
    for (const auto& fs: inFastSeqs)
      inSeqs.push_back (NamedInputSeq ({ fs.name, splitToChars (fs.seq) }));
    for (const auto& fs: outFastSeqs)
      outSeqs.push_back (NamedOutputSeq ({ fs.name, splitToChars (fs.seq) }));
    if (vm.count("input-json"))
      inSeqs.push_back (JsonReader<NamedInputSeq>::fromFile (vm.at("input-json").as<string>()));
    if (vm.count("output-json"))
      outSeqs.push_back (JsonReader<NamedOutputSeq>::fromFile (vm.at("output-json").as<string>()));
    
    // if inputs/outputs specified individually, create all input-output pairs
    if (inSeqs.empty() && ((!outSeqs.empty() && machine.inputAlphabet().empty()) || vm.count("prefix-encode") || vm.count("beam-encode") || vm.count("random-encode") || vm.count("prefix-decode") || vm.count("cool-decode") || vm.count("mcmc-decode") || vm.count("beam-decode")))
      inSeqs.push_back (NamedInputSeq());  // create a dummy input if we have outputs & either the input alphabet is empty, or we're encoding/decoding
    if (outSeqs.empty() && ((!inSeqs.empty() && machine.outputAlphabet().empty()) || vm.count("prefix-encode") || vm.count("beam-encode") || vm.count("random-encode")))
      outSeqs.push_back (NamedOutputSeq());  // create a dummy output if the output alphabet is empty, or we're encoding
    for (const auto& inSeq: inSeqs)
      for (const auto& outSeq: outSeqs)
	data.seqPairs.push_back (SeqPair ({ inSeq, outSeq }));

    // after all that, do we have data? did we need data?
    const bool noIO = machine.inputAlphabet().empty() && machine.outputAlphabet().empty();
    if (inferenceRequested && data.seqPairs.empty() && noIO)
      data.seqPairs.push_back (SeqPair());  // if the model has no I/O, then add an automatic pair of empty, nameless sequences (the only possible evidence)
    const bool gotData = !data.seqPairs.empty();
    Require (!gotData || inferenceRequested, "No point in specifying input/output data without --train, --loglike, --counts, --align, --encode, or --decode");

    // fit parameters
    Params params;
    if (vm.count("train")) {
      Require ((vm.count("constraints") || !machine.cons.empty())
	       && (gotData || noIO),
	       "To fit parameters, please specify a constraints file and (for machines with input/output) a data file");
      MachineFitter fitter;
      fitter.machine = machine;
      if (vm.count("constraints"))
	fitter.constraints = constraints;
      fitter.constants = funcs;
      fitter.seed = fitter.allConstraints().defaultParams().combine (seed, true);
      params = vm.count("wiggle-room") ? fitter.fit(data,vm.at("wiggle-room").as<int>()) : fitter.fit(data);
      cout << JsonLoader<Params>::toJsonString(params) << endl;
    } else
      params = funcs.combine (seed).combine (machine.getParamDefs (vm.count("use-defaults")));

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

    // random seed
    auto makeRnd = [&] () -> mt19937 {
      time_t timer;
      time (&timer);
      const int seed = vm.count("seed") ? vm.at("seed").as<int>() : timer;
      LogThisAt(2,"Random seed is " << seed << endl);
      mt19937 mt (seed);
      return mt;
    };
    
    // encode
    const long maxBacktrack = vm.count("backtrack") ? vm.at("backtrack").as<long>() : numeric_limits<long>::max();
    if (vm.count("prefix-encode") || vm.count("beam-encode") || vm.count("random-encode")) {
      Require (gotData, "To encode an output sequence, please specify an input sequence file");
      const Machine trans = machine.transpose().advanceSort().advancingMachine();
      const Machine decodeTrans = vm.count("beam-encode") ? trans.decodeSort() : trans;
      const EvaluatedMachine eval (decodeTrans, params);
      SeqPairList encodeResults;
      for (const auto& seqPair: data.seqPairs) {
	Require (seqPair.output.seq.size() == 0, "You cannot specify output sequences when encoding; the goal of encoding is to generate %s output for a given input", vm.count("random-encode") ? "random" : "the most likely");
	vguard<OutputSymbol> encoded;
	if (vm.count("beam-encode")) {
	  if (!decodeTrans.isDecodingMachine())
	    Warn ("Machine is not topologically sorted for encoding; some valid outputs may be missed");
	  const size_t beamWidth = vm.count("beam-width") ? vm.at("beam-width").as<size_t>() : DefaultBeamWidth;
	  BeamSearchMatrix beam (eval, seqPair.input.seq, beamWidth);
	  encoded = beam.bestSeq();
	} else {
	  PrefixTree tree (eval, (vguard<OutputSymbol>) seqPair.input.seq, maxBacktrack);
	  if (vm.count("random-encode")) {
	    mt19937 mt = makeRnd();
	    encoded = tree.sampleSeq (mt);
	  } else
	    encoded = tree.doPrefixSearch();
	}
	encodeResults.seqPairs.push_back (SeqPair ({ seqPair.input, NamedOutputSeq ({ string(DefaultOutputSequenceName), encoded }) }));
      }
      encodeResults.writeJson (cout);
      cout << endl;
    }

    // decode
    if (vm.count("prefix-decode") || vm.count("cool-decode") || vm.count("mcmc-decode") || vm.count("beam-decode")) {
      Require (gotData, "To decode an input sequence, please specify an output sequence file");
      const Machine decodeMachine = vm.count("beam-decode") ? machine.decodeSort() : machine;
      const EvaluatedMachine eval (decodeMachine, params);
      SeqPairList decodeResults;
      for (const auto& seqPair: data.seqPairs) {
	Require (seqPair.input.seq.size() == 0, "You cannot specify input sequences when decoding; the goal of decoding is to impute the most likely input for a given output");
	vguard<InputSymbol> decoded;
	if (vm.count("beam-decode")) {
	  if (!decodeMachine.isDecodingMachine())
	    Warn ("Machine is not topologically sorted for decoding; some valid inputs may be missed");
	  const size_t beamWidth = vm.count("beam-width") ? vm.at("beam-width").as<size_t>() : DefaultBeamWidth;
	  BeamSearchMatrix beam (eval, seqPair.output.seq, beamWidth);
	  decoded = beam.bestSeq();
	} else {
	  PrefixTree tree (eval, seqPair.output.seq, maxBacktrack);
	  if (vm.count("cool-decode") || vm.count("mcmc-decode")) {
	    mt19937 mt = makeRnd();
	    const int defaultSteps = 10;
	    decoded = tree.doAnnealedSearch (mt, vm.count("decode-steps") ? vm.at("decode-steps").as<int>() : defaultSteps, vm.count("cool-decode"));
	  } else
	    decoded = tree.doPrefixSearch();
	}
	decodeResults.seqPairs.push_back (SeqPair ({ NamedInputSeq ({ string(DefaultInputSequenceName), decoded }), seqPair.output }));
      }
      decodeResults.writeJson (cout);
      cout << endl;
    }
    
  } catch (const std::exception& e) {
    cerr << e.what() << endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}
