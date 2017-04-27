#include "caller.h"
#include "../logger.h"

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

void BaseCallingParams::init (const string& alph, SeqIdx len, int cpts) {
  alphabet = alph;
  kmerLen = len;
  components = cpts;
  params.gauss.clear();
  params.prob.clear();
  const Kmer nk = numberOfKmers (len, alph.size());
  for (Kmer kmer = 0; kmer < nk; ++kmer) {
    const string kmerStr = kmerToString (kmer, len, alph);
    const string prefix = kmerStr.substr(0,kmerLen-1);
    const char suffix = kmerStr[kmerLen-1];
    GaussianParams emit;
    for (int cpt = 0; cpt < cpts; ++cpt) {
      params.prob.defs[cptWeightLabel (kmerStr, cpt)] = 1. / (double) cpts;
      params.prob.defs[cptExtendLabel (kmerStr, cpt)] = .5;
      params.prob.defs[cptEndLabel (kmerStr, cpt)] = .5;
    }
    params.prob.defs[condFreqLabel (prefix, suffix)] = 1. / (double) alph.size();
    params.gauss[emitLabel (kmerStr)] = GaussianParams();
  }
}

json BaseCallingParams::asJson() const {
  return json::object ({ { "alphabet", alphabet }, { "kmerlen", kmerLen }, { "components", components }, { "params", params.asJson() } });
}

void BaseCallingParams::writeJson (ostream& out) const {
  out << asJson();
}

void BaseCallingParams::readJson (const json& j) {
  alphabet = j["alphabet"].get<string>();
  kmerLen = j["kmerlen"].get<int>();
  components = j["components"].get<int>();
  params.readJson (j["params"]);
}

BaseCallingPrior::BaseCallingPrior()
  : condFreq(1),
    cptWeight(1),
    cptExtend(1),
    cptEnd(1),
    mu(0), muCount(1),
    tau(1), tauCount(2)
{ }

GaussianModelPrior BaseCallingPrior::modelPrior (const string& alph, SeqIdx kmerLen, int components) const {
  GaussianModelPrior prior;
  
  GaussianPrior emitPrior;
  emitPrior.mu0 = mu;
  emitPrior.n_mu = muCount;
  emitPrior.tau0 = tau;
  emitPrior.n_tau = tauCount;

  const Kmer nk = numberOfKmers (kmerLen, alph.size());
  for (Kmer kmerPrefix = 0; kmerPrefix < nk; kmerPrefix += alph.size()) {
    vguard<string> condFreqParam;
    condFreqParam.reserve (alph.size());
    for (Kmer kmer = kmerPrefix; kmer < kmerPrefix + alph.size(); ++kmer) {
      const string kmerStr = kmerToString (kmer, kmerLen, alph);
      const string prefix = kmerStr.substr(0,kmerLen-1);
      const char suffix = kmerStr[kmerLen-1];
      vguard<string> cptWeightParam;
      cptWeightParam.reserve (components);
      for (int cpt = 0; cpt < components; ++cpt) {
	prior.count.defs[cptWeightLabel (kmerStr, cpt)] = cptWeight;
	prior.count.defs[cptExtendLabel (kmerStr, cpt)] = cptExtend;
	prior.count.defs[cptEndLabel (kmerStr, cpt)] = cptEnd;
	prior.cons.norm.push_back (vguard<string> { cptExtendLabel (kmerStr, cpt), cptEndLabel (kmerStr, cpt) });
	cptWeightParam.push_back (cptWeightLabel (kmerStr, cpt));
      }
      prior.cons.norm.push_back (cptWeightParam);
      prior.count.defs[condFreqLabel (prefix, suffix)] = condFreq;
      prior.gauss[emitLabel (kmerStr)] = emitPrior;
      condFreqParam.push_back (condFreqLabel (prefix, suffix));
    }
    prior.cons.norm.push_back (condFreqParam);
  }

  return prior;
}

void BaseCallingMachine::init (const string& alph, SeqIdx len, int cpts) {
  components = cpts;
  nKmers = numberOfKmers (len, alph.size());
  state = vguard<MachineState> (nKmers * (components + 2) + 2);
  state[startState()].name = "start";
  state[endState()].name = "end";
  for (Kmer kmer = 0; kmer < nKmers; ++kmer) {
    const string kmerStr = kmerToString (kmer, len, alph);
    const string suffix = kmerStr.substr(1);
    MachineState& start (state[kmerStart(kmer)]);
    MachineState& end (state[kmerEnd(kmer)]);
    start.name = kmerStr + "_start";
    end.name = kmerStr + "_end";
    for (int cpt = 0; cpt < cpts; ++cpt) {
      MachineState& sc = state[kmerEmit(kmer,cpt)];
      sc.name = kmerStr + "_" + cptName(cpt);
      start.trans.push_back (MachineTransition (string(), emitLabel(kmerStr), kmerEmit(kmer,cpt), WeightExpr (cptWeightLabel (kmerStr, cpt))));
      sc.trans.push_back (MachineTransition (string(), emitLabel(kmerStr), kmerEmit(kmer,cpt), WeightExpr (cptExtendLabel (kmerStr, cpt))));
      sc.trans.push_back (MachineTransition (string(), string(), kmerEnd(kmer), WeightExpr (cptEndLabel (kmerStr, cpt))));
    }
    for (auto c: alph)
      end.trans.push_back (MachineTransition (string(1,c), string(), kmerStart(stringToKmer(suffix+c,alph)), WeightExpr (condFreqLabel (suffix, c))));
    state[startState()].trans.push_back (MachineTransition (string(), string(), kmerEnd(kmer), WeightExpr(1. / (double) nKmers)));
    end.trans.push_back (MachineTransition (string(), string(), nStates() - 1, WeightExpr(true)));
  }
}
