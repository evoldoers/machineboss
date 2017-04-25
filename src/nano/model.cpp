#include "caller.h"
#include "logger.h"

string BaseCallingParamNamer::emitLabel (const string& kmerStr) {
  return string("emit(") + kmerStr + ")";
}

string BaseCallingParamNamer::condFreqLabel (const string& prefix, const char suffix) {
  return string("P(") + suffix + "|" + prefix + ")";
}

string BaseCallingParamNamer::cptWeightLabel (const string& kmerStr, int cpt) {
  return string("P(") + cptName(cpt) + "|" + kmerStr + ")";
}

string BaseCallingParamNamer::cptExtendLabel (const string& kmerStr, int cpt) {
  return string("P(ext|") + kmerStr + "," + cptName(cpt) + ")";
}

string BaseCallingParamNamer::cptEndLabel (const string& kmerStr, int cpt) {
  return string("P(end|") + kmerStr + "," + cptName(cpt) + ")";
}

string BaseCallingParamNamer::cptName (int cpt) {
  return string("cpt") + to_string(cpt + 1);
}

BaseCallingParams::init (const string& alph, SeqIndex len, int cpts) {
  const Kmer nk = numberOfKmers (len, alph.size());
  for (KmerIndex kmer = 0; kmer < nk; ++kmer) {
    const string kmerStr = kmerToString (kmer, len, alph);
    const string prefix = kmerStr.substr(0,kmerLen-1);
    const char suffix = kmerStr[kmerLen-1];
    GaussianParams emit;
    for (int cpt = 0; cpt < cpts; ++cpt) {
      prob[cptWeightLabel (kmerStr, cpt)] = 1. / (double) cpts;
      prob[cptExtendLabel (kmerStr, cpt)] = .5;
      prob[cptEndLabel (kmerStr, cpt)] = .5;
    }
    prob[condFreqLabel (prefix, suffix)] = 1. / (double) alph.size();
    gauss[emitLabel (kmerStr)] = GaussianParams();
  }
}

BaseCallingPrior::BaseCallingPrior()
  : condFreq(1),
    cptWeight(1),
    cptExtend(1),
    cptEnd(1),
    mu(0), muCount(1),
    tau(1), tauCount(2)
{ }

ModelPrior BaseCallingPrior::modelPrior (const string& alph, SeqIndex kmerLen, int components) const {
  ModelPrior prior;
  
  GaussianPrior emitPrior;
  emitPrior.mu0 = mu;
  emitPrior.n_mu = muCount;
  emitPrior.tau0 = tau;
  emitPrior.n_tau = tauCount;

  const Kmer nk = numberOfKmers (kmerLen, alph.size());
  for (Kmer kmerPrefix = 0; kmerPrefix < nk; kmer += alph.size()) {
    vguard<string> condFreqParam;
    condFreqParam.reserve (alph.size());
    for (Kmer kmer = kmerPrefix; kmer < kmerPrefix + alph.size(); ++kmer) {
      const string kmerStr = kmerToString (kmer, len, alph);
      const string prefix = kmerStr.substr(0,kmerLen-1);
      const char suffix = kmerStr[kmerLen-1];
      vguard<string> cptWeightParam;
      cptWeightParam.reserve (cpts);
      for (int cpt = 0; cpt < cpts; ++cpt) {
	prior.count[cptWeightLabel (kmerStr, cpt)] = cptWeight;
	prior.count[cptExtendLabel (kmerStr, cpt)] = cptExtend;
	prior.count[cptEndLabel (kmerStr, cpt)] = cptEnd;
	prior.norm.push_back (vguard<string> { cptExtendLabel (kmerStr, cpt), cptEndLabel (kmerStr, cpt) });
	cptWeightParam.push_back (cptWeightLabel (kmerStr, cpt));
      }
      prior.norm.push_back (cptWeightParam);
      prior.count[condFreqLabel (prefix, suffix)] = condFreq;
      prior.gauss[emitLabel (kmerStr)] = emitPrior;
      condFreqParam.push_back (condFreqLabel (prefix, suffix));
    }
    prior.norm.push_back (condFreqParam);
  }

  return prior;
}

void BaseCallingModel::init (const string& alph, SeqIndex len, int cpts) {
  const Kmer nk = numberOfKmers (len, alph.size());
  state = vguard<MachineState> (nk * (cpts + 2) + 2);
  state[startState()].name = "start";
  state[endState()].name = "end";
  for (KmerIndex kmer = 0; kmer < nk; ++kmer) {
    const StateIndex si = kmerStart (kmer);
    const string kmerStr = kmerToString (kmer, len, alph);
    const string suffix = kmerStr.substr(1);
    State& start (state[si]), end (state[si + cpts + 1]);
    start.name = kmerStr + "_start";
    end.name = kmerStr + "_end";
    for (int cpt = 0; cpt < cpts; ++cpt) {
      State& sc = state[si + cpt + 1];
      sc.name = kmerStr + "_" + cptName[cpt];
      start.outgoing.push_back (MachineTransition (string(), emitLabel(kmerStr), si + cpt + 1, WeightExpr (cptWeightLabel (kmerStr, cpt))));
      sc.outgoing.push_back (MachineTransition (string(), emitLabel(kmerStr), si + cpt + 1, WeightExpr (cptExtendLabel (kmerStr, cpt))));
      sc.outgoing.push_back (MachineTransition (string(), string(), si + cpts + 1, WeightExpr (cptEndLabel (kmerStr, cpt))));
    }
    for (auto c: alph)
      end.outgoing.push_back (MachineTransition (string(1,c), string(), kmerStart (stringToKmer (suffix + c, alph)), WeightExpr (condFreqLabel (suffix, c))));
    state[startState()].outgoing.push_back (MachineTransition (string(), string(), si, WeightExpr(1. / (double) nk)));
    end.outgoing.push_back (MachineTransition (string(), string(), si, WeightExpr(1.)));
  }
}
