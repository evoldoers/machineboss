#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <random>
#include <deque>
#include <boost/program_options.hpp>

#include "../src/vguard.h"
#include "../src/logger.h"
#include "../src/util.h"
#include "../src/jsonio.h"
#include "../src/fastseq.h"
#include "../src/nano/trace.h"
#include "../src/nano/caller.h"
#include "../src/nano/gtrainer.h"

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

    po::options_description modelOpts("Model options");
    modelOpts.add_options()
      ("alphabet,a", po::value<string>()->default_value("acgt"), "alphabet")
      ("kmerlen,k", po::value<int>()->default_value(6), "kmer length")
      ("components,c", po::value<int>()->default_value(1), "# of mixture components in length distributions")
      ;

    po::options_description dpOpts("Event detection & DP options");
    dpOpts.add_options()
      ("maxfracdiff,f", po::value<double>()->default_value(.01), "max fractional delta between samples in same event")
      ("maxsamples,s", po::value<size_t>()->default_value(4), "max number of samples per event")
      ("memlimit,m", po::value<size_t>()->default_value(1<<30), "approximate memory limit for forward-backward DP")
      ("bandwidth,w", po::value<double>()->default_value(1), "proportion of DP matrix to fill around main diagonal")
      ;

    po::options_description appOpts("Data options");
    appOpts.add_options()
      ("fast5,D", po::value<vector<string> >(), "load trace data from FAST5 file(s)")
      ("raw,R", po::value<vector<string> >(), "load trace from text file(s) e.g. from 'f5dump --rw' in fast5")
      ("model,M", po::value<string>(), "load model parameters from file")
      ("fasta,T", po::value<vector<string> >(), "load training sequences from FASTA/FASTQ file(s), then fit using EM")
      ("save,S", po::value<string>(), "save trained model parameters to file")
      ("call,C", "base-call using Viterbi decoding, first fitting scaling parameters using EM")
      ;

    po::positional_options_description posOpts;
    posOpts.add("data", -1);

    po::options_description parseOpts("");
    parseOpts.add(generalOpts).add(modelOpts).add(appOpts).add(dpOpts);

    // parse args
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc,argv).options(parseOpts).positional(posOpts).run();
    po::store (parsed, vm);

    // handle help
    if (vm.count("help")) {
      cout << parseOpts << endl;
      return 1;
    }

    // test option validity
    po::notify(vm);

    // set logging options
    logger.parseLogArgs (vm);

    // load parameters
    LogThisAt(2,"Initializing model" << endl);
    BaseCallingParams initParams;
    if (vm.count("model"))
      JsonReader<BaseCallingParams>::readFile (initParams, vm.at("model").as<string>());
    else
      initParams.init (vm.at("alphabet").as<string>(),
		       vm.at("kmerlen").as<int>(),
		       vm.at("components").as<int>());

    // read data
    Require (vm.count("fast5") || vm.count("raw"), "Please specify at least one data file");
    LogThisAt(2,"Reading trace data" << endl);
    TraceList traceList;
    if (vm.count("fast5"))
      for (const auto& fast5Filename: vm.at("fast5").as<vector<string> >())
	traceList.readFast5 (fast5Filename);
    if (vm.count("raw"))
      for (const auto& textFilename: vm.at("raw").as<vector<string> >())
	traceList.readText (textFilename);

    for (auto& trace: traceList.trace)
      trace.normalize();
    
    // segment
    const TraceMomentsList traceMomentsList (traceList, vm.at("maxfracdiff").as<double>(), vm.at("maxsamples").as<size_t>());
    
    // initialize machine & prior
    BaseCallingMachine machine;
    machine.init (initParams.alphabet, initParams.kmerLen, initParams.components);

    BaseCallingPrior bcPrior;
    const GaussianModelPrior modelPrior = bcPrior.modelPrior (initParams.alphabet, initParams.kmerLen, initParams.components);
    
    // train model
    BaseCallingParams trainedParams = initParams;
    if (vm.count("fasta")) {
      LogThisAt(2,"Reading sequence data" << endl);
      vguard<FastSeq> trainSeqs;
      for (const auto& seqFilename: vm.at("fasta").as<vector<string> >())
	readFastSeqs (seqFilename.c_str(), trainSeqs);

      GaussianModelFitter fitter;
      fitter.init (machine, initParams.params, modelPrior, traceMomentsList, trainSeqs);
      fitter.blockBytes = vm.at("memlimit").as<size_t>() / 2;
      fitter.bandWidth = vm.at("bandwidth").as<double>();
      fitter.fit();

      trainedParams.params = fitter.modelParams;
    }
    
    // save parameters
    if (vm.count("save"))
      JsonWriter<BaseCallingParams>::toFile (trainedParams, vm.at("save").as<string>());
    else if (!vm.count("call"))
      trainedParams.writeJson (cout);

    // do basecalling
    if (vm.count("call")) {
      GaussianDecoder decoder;
      decoder.init (machine, trainedParams.params, modelPrior, traceMomentsList);
      writeFastaSeqs (cout, decoder.decode());
    }

  } catch (const std::exception& e) {
    cerr << e.what() << endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}
