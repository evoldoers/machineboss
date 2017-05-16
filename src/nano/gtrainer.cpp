#include "gtrainer.h"
#include "fwdtrace.h"
#include "backtrace.h"
#include "vtrace.h"
#include "../logger.h"

#define MaxEMIterations 1000
#define MinEMImprovement .001

GaussianTrainer::GaussianTrainer() :
  blockBytes(0),
  bandWidth(1)
{ }

void GaussianTrainer::init (const EventMachine& em, const GaussianModelParams& mp, const GaussianModelPrior& pr, const TraceMomentsList& tl) {
  eventMachine = em;
  prior = pr;
  modelParams = mp;
  traceList = tl;
  traceListParams.init (traceList);
}

void GaussianTrainer::reset() {
  LogThisAt(7,"Model parameters, iteration #" << (iter+1) << ":" << endl << JsonWriter<GaussianModelParams>::toJsonString(modelParams));
  LogThisAt(7,"Training set parameters, iteration #" << (iter+1) << ":" << endl << JsonWriter<TraceListParams>::toJsonString(traceListParams));
  logPrior = prior.logProb (modelParams, traceListParams);
  logLike = logPrior;
  counts.clear();
}

bool GaussianTrainer::testFinished() {
  LogThisAt(2,"Baum-Welch, iteration #" << (iter+1) << ": log-likelihood " << logLike << endl);
  LogThisAt(3,"Log-prior, iteration #" << (iter+1) << ": " << logPrior << endl);
  LogThisAt(4,"Expected log-likelihood (emissions) before M-step: " << expectedLogLike() << endl);
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

double GaussianTrainer::expectedLogLike() const {
  return GaussianModelCounts::expectedLogLike (eventMachine, modelParams, traceListParams, prior, counts);
}

void GaussianModelFitter::init (const EventMachine& em, const GaussianModelParams& mp, const GaussianModelPrior& pr, const TraceMomentsList& tl, const vguard<FastSeq>& s) {
  GaussianTrainer::init (em, mp, pr, tl);
  seqs = s;
  inputConditionedMachine.clear();
  for (auto& fs: seqs) {
    vguard<OutputSymbol> seq (fs.length());
    for (SeqIdx pos = 0; pos < fs.length(); ++pos)
      seq[pos] = string (1, tolower (fs.seq[pos]));
    inputConditionedMachine.push_back (Machine::compose (Machine::generator (fs.name, seq), eventMachine.machine, false, false).eliminateSilentTransitions());
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
      const Machine& machine = *(machineIter++);
      const EvaluatedMachine eval (machine, modelParams.params().combine (eventMachine.event));
      GaussianModelCounts c;
      c.init (eval);
      logLike += c.add (machine, eventMachine.event, eval, modelParams, trace, traceParams, blockBytes, bandWidth);
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
      (*(countIter++)).optimizeTraceParams (traceParams, eventMachine, *(evalIter++), modelParams, prior);
    LogThisAt(4,"Expected log-likelihood (emissions) after optimizing trace parameters: " << expectedLogLike() << endl);

    GaussianModelCounts::optimizeModelParams (modelParams, traceListParams, prior, eventMachine, evalMachine, counts);
    LogThisAt(4,"Expected log-likelihood (emissions) after optimizing model parameters: " << expectedLogLike() << endl);
  }
}

vguard<FastSeq> GaussianDecoder::decode() {
  vguard<FastSeq> result;
  size_t m = 0;
  for (const auto& trace: traceList.trace) {
    LogThisAt(3,"Fitting scaling parameters for trace " << trace.name << endl);
    const EvaluatedMachine eval (eventMachine.machine, modelParams.params().combine (eventMachine.event));
    TraceParams& traceParams = traceListParams.params[m];
    for (iter = 0; true; ++iter) {
      reset();
      GaussianModelCounts c;
      c.init (eval);
      logLike += c.add (eventMachine.machine, eventMachine.event, eval, modelParams, trace, traceParams, blockBytes, bandWidth);
      counts.push_back (c);
      if (testFinished())
	break;
      c.optimizeTraceParams (traceParams, eventMachine, eval, modelParams, prior);
      LogThisAt(4,"Expected log-likelihood after optimizing trace parameters: " << expectedLogLike() << endl);
    }
    ViterbiTraceMatrix viterbi (eval, modelParams, trace, traceParams);
    FastSeq fs;
    fs.name = trace.name;
    for (const auto& trans: viterbi.path(eventMachine.machine).trans)
      if (!trans.inputEmpty())
	fs.seq.append (trans.in);
    result.push_back (fs);
    ++m;
  }
  return result;
}
