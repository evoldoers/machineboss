#include "gtrainer.h"
#include "fwdtrace.h"
#include "backtrace.h"
#include "vtrace.h"
#include "../logger.h"

#define MaxEMIterations 1000
#define MinEMImprovement .001

void GaussianTrainer::init (const Machine& m, const GaussianModelParams& mp, const GaussianModelPrior& pr, const TraceList& tl) {
  machine = m;
  prior = pr;
  modelParams = mp;
  traceList = tl;
  traceListParams.init (traceList);
}

void GaussianTrainer::reset() {
  LogThisAt(5,"Model parameters, iteration #" << (iter+1) << ":" << endl << JsonWriter<GaussianModelParams>::toJsonString(modelParams));
  LogThisAt(5,"Training set parameters, iteration #" << (iter+1) << ":" << endl << JsonWriter<TraceListParams>::toJsonString(traceListParams));
  logPrior = prior.logProb (modelParams, traceListParams);
  logLike = logPrior;
  counts.clear();
}

bool GaussianTrainer::testFinished() {
  LogThisAt(2,"Baum-Welch, iteration #" << (iter+1) << ": log-likelihood " << logLike << endl);
  LogThisAt(3,"Log-prior, iteration #" << (iter+1) << ": " << logPrior << endl);
  LogThisAt(4,"Expected log-likelihood (emissions) before M-step: " << expectedLogEmit() << endl);
  if (iter > 0) {
    if (iter == MaxEMIterations) {
      LogThisAt(2,"Reached " << MaxEMIterations << " iterations; stopping" << endl);
      return true;
    }
    const double improvement = (logLike - prevLogLike) / abs(prevLogLike);
    if (improvement < MinEMImprovement) {
      LogThisAt(2,"Fractional improvement (" << improvement << ") is below threshold (" << MinEMImprovement << "); stopping" << endl);
      return true;
    }
  }
  prevLogLike = logLike;
  return false;
}

double GaussianTrainer::expectedLogEmit() const {
  return GaussianModelCounts::expectedLogEmit (modelParams, traceListParams, prior, counts);
}

void GaussianModelFitter::init (const Machine& m, const GaussianModelParams& mp, const GaussianModelPrior& pr, const TraceList& tl, const vguard<FastSeq>& s) {
  GaussianTrainer::init (m, mp, pr, tl);
  seqs = s;
  inputConditionedMachine.clear();
  for (auto& fs: seqs) {
    vguard<OutputSymbol> seq (fs.length());
    for (SeqIdx pos = 0; pos < fs.length(); ++pos)
      seq[pos] = string (1, fs.seq[pos]);
    inputConditionedMachine.push_back (Machine::compose (Machine::generator (fs.name, seq), machine));
  }
}

void GaussianModelFitter::fit() {
  for (iter = 0; true; ++iter) {
    reset();
    size_t m = 0;
    list<EvaluatedMachine> evalMachine;
    auto machineIter = inputConditionedMachine.begin();
    for (const auto& trace: traceList.trace) {
      const TraceParams& traceParams = traceListParams.params[m];
      const EvaluatedMachine eval (*(machineIter++), modelParams.prob);
      GaussianModelCounts c;
      c.init (eval);
      logLike += c.add (eval, modelParams, trace, traceParams);
      counts.push_back (c);
      evalMachine.push_back (eval);
      LogThisAt(6,"Counts for trace #" << m << ", iteration #" << (iter+1) << ":" << endl << JsonWriter<GaussianModelCounts>::toJsonString(c) << endl);
      ++m;
    }
    if (testFinished())
      break;

    auto countIter = counts.begin();
    auto evalIter = evalMachine.begin();
    for (auto& traceParams: traceListParams.params)
      (*(countIter++)).optimizeTraceParams (traceParams, *(evalIter++), modelParams, prior);
    LogThisAt(4,"Expected log-likelihood (emissions) after optimizing trace parameters: " << expectedLogEmit() << endl);

    GaussianModelCounts::optimizeModelParams (modelParams, traceListParams, prior, inputConditionedMachine, evalMachine, counts);
    LogThisAt(4,"Expected log-likelihood (emissions) after optimizing model parameters: " << expectedLogEmit() << endl);
  }
}

vguard<FastSeq> GaussianDecoder::decode() {
  vguard<FastSeq> result;
  size_t m = 0;
  for (const auto& trace: traceList.trace) {
    LogThisAt(3,"Fitting scaling parameters for trace " << trace.name << endl);
    const EvaluatedMachine eval (machine, modelParams.prob);
    TraceParams& traceParams = traceListParams.params[m];
    for (iter = 0; true; ++iter) {
      reset();
      GaussianModelCounts c;
      c.init (eval);
      logLike += c.add (eval, modelParams, trace, traceParams);
      counts.push_back (c);
      if (testFinished())
	break;
      c.optimizeTraceParams (traceParams, eval, modelParams, prior);
      LogThisAt(4,"Expected log-likelihood after optimizing trace parameters: " << expectedLogEmit() << endl);
    }
    ViterbiTraceMatrix viterbi (eval, modelParams, trace, traceParams);
    FastSeq fs;
    fs.name = trace.name;
    for (const auto& trans: viterbi.path(machine).trans)
      if (!trans.inputEmpty())
	fs.seq.append (trans.in);
    result.push_back (fs);
    ++m;
  }
  return result;
}
