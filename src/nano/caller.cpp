#include "caller.h"
#include "../logger.h"

string BaseCallingParamNamer::padEmitLabel() {
  return string("padEmit");
}

string BaseCallingParamNamer::padExtendLabel() {
  return string("padExtend");
}

string BaseCallingParamNamer::padEndLabel() {
  return string("padEnd");
}

string BaseCallingParamNamer::emitLabel (const string& kmerStr) {
  return string("emit(") + kmerStr + ")";
}

string BaseCallingParamNamer::cptWeightLabel (const string& kmerStr, int cpt) {
  return string("P(") + cptName(cpt) + "|" + kmerStr + ")";
}

string BaseCallingParamNamer::cptExtendLabel (const string& kmerStr, int cpt) {
  return waitEventFuncName (cptExitRateLabel (kmerStr, cpt));
}

string BaseCallingParamNamer::cptEndLabel (const string& kmerStr, int cpt) {
  return exitEventFuncName (cptExitRateLabel (kmerStr, cpt));
}

string BaseCallingParamNamer::cptExitRateLabel (const string& kmerStr, int cpt) {
  return string("R(move|") + kmerStr + "," + cptName(cpt) + ")";
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
      if (cpts > 1)
	params.prob.defs[cptWeightLabel (kmerStr, cpt)] = 1. / (double) cpts;
      params.rate.defs[cptExitRateLabel (kmerStr, cpt)] = 1. / (cpt + 1);
    }
    params.gauss[emitLabel (kmerStr)] = GaussianParams();
  }
  params.gauss[padEmitLabel()] = GaussianParams();
  params.prob.defs[padExtendLabel()] = .5;
  params.prob.defs[padEndLabel()] = .5;
}

json BaseCallingParams::asJson() const {
  return json::object ({ { "alphabet", alphabet }, { "kmerlen", kmerLen }, { "components", components }, { "params", params.asJson() } });
}

void BaseCallingParams::writeJson (ostream& out) const {
  out << asJson() << endl;
}

void BaseCallingParams::readJson (const json& j) {
  alphabet = j["alphabet"].get<string>();
  kmerLen = j["kmerlen"].get<int>();
  components = j["components"].get<int>();
  params.readJson (j["params"]);
}

BaseCallingPrior::BaseCallingPrior()
  : cptWeight(1),
    padExtend(1),
    padEnd(1),
    cptExitCount(1),
    cptExitTime(1),
    mu (BaseCallerMuMean),
    muCount (CALC_N_MU (BaseCallerMuError, CALC_TAU_MEAN(BaseCallerSigmaMean), CALC_TAU_SD(BaseCallerSigmaError))),
    tau (CALC_TAU_MODE (CALC_TAU_MEAN(BaseCallerSigmaMean), CALC_TAU_SD(BaseCallerSigmaError))),
    tauCount (CALC_N_TAU (CALC_TAU_MEAN(BaseCallerSigmaMean), CALC_TAU_SD(BaseCallerSigmaError))),
    muPad (BaseCallerPadMuMean),
    muPadCount (CALC_N_MU (BaseCallerPadMuError, CALC_TAU_MEAN(BaseCallerPadSigmaMean), CALC_TAU_SD(BaseCallerPadSigmaError))),
    tauPad (CALC_TAU_MODE (CALC_TAU_MEAN(BaseCallerPadSigmaMean), CALC_TAU_SD(BaseCallerPadSigmaError))),
    tauPadCount (CALC_N_TAU (CALC_TAU_MEAN(BaseCallerPadSigmaMean), CALC_TAU_SD(BaseCallerPadSigmaError)))
{ }

GaussianModelPrior BaseCallingPrior::modelPrior (const string& alph, SeqIdx kmerLen, int components) const {
  GaussianModelPrior prior;
  (TraceParamsPrior&) prior = *this;

  GaussianPrior emitPrior;
  emitPrior.mu0 = mu;
  emitPrior.n_mu = muCount;
  emitPrior.tau0 = tau;
  emitPrior.n_tau = tauCount;

  GammaPrior exitPrior;
  exitPrior.count = cptExitCount;
  
  const Kmer nk = numberOfKmers (kmerLen, alph.size());
  for (Kmer kmerPrefix = 0; kmerPrefix < nk; kmerPrefix += alph.size())
    for (Kmer kmer = kmerPrefix; kmer < kmerPrefix + alph.size(); ++kmer) {
      const string kmerStr = kmerToString (kmer, kmerLen, alph);
      const string prefix = kmerStr.substr(0,kmerLen-1);
      const char suffix = kmerStr[kmerLen-1];
      vguard<string> cptWeightParam;
      cptWeightParam.reserve (components);
      for (int cpt = 0; cpt < components; ++cpt) {
	if (components > 1)
	  prior.count.defs[cptWeightLabel (kmerStr, cpt)] = cptWeight;
	exitPrior.time = cptExitTime * (cpt + 1);
	prior.gamma[cptExitRateLabel (kmerStr, cpt)] = exitPrior;
	prior.cons.rate.push_back (cptExitRateLabel (kmerStr, cpt));
	cptWeightParam.push_back (cptWeightLabel (kmerStr, cpt));
      }
      if (components > 1)
	prior.cons.norm.push_back (cptWeightParam);
      prior.gauss[emitLabel (kmerStr)] = emitPrior;
    }

  GaussianPrior padPrior;
  padPrior.mu0 = muPad;
  padPrior.n_mu = muCount;
  padPrior.tau0 = tauPad;
  padPrior.n_tau = tauCount;

  prior.gauss[padEmitLabel()] = padPrior;
  prior.count.defs[padExtendLabel()] = padExtend;
  prior.count.defs[padEndLabel()] = padEnd;
  prior.cons.norm.push_back (vguard<string> { padExtendLabel(), padEndLabel() });

  return prior;
}

void BaseCallingMachine::init (const string& alph, SeqIdx len, int cpts) {
  alphSize = alph.size();
  components = cpts;
  nKmers = numberOfKmers (len, alphSize);
  kmerOffset = nShorterKmers (len);
  state = vguard<MachineState> (nKmers * (components + 2) + kmerOffset + 1);
  state[startState()].name = "start";
  state[endState()].name = "end";

  for (int shortLen = 0; shortLen < len; ++shortLen) {
    const Kmer nShort = numberOfKmers (shortLen, alphSize);
    for (Kmer sk = 0; sk < nShort; ++sk) {
      MachineState& skState (state[shortKmer(sk,shortLen)]);
      const string suffix = kmerToString (sk, shortLen, alph);
      if (shortLen > 0)
	skState.name = suffix;
      for (AlphTok nextTok = 0; nextTok < alphSize; ++nextTok) {
	const Kmer nk = sk * alphSize + nextTok;
	const char c = alph[nextTok];
	const StateIndex nki = (shortLen + 1 == len) ? kmerStart(nk) : shortKmer(nk,shortLen+1);
	skState.trans.push_back (MachineTransition (string(1,c), string(), nki, WeightExpr(true)));
      }
    }
  }

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
      start.trans.push_back (MachineTransition (string(), emitLabel(kmerStr), kmerEmit(kmer,cpt), cpts == 1 ? WeightExpr(true) : WeightExpr (cptWeightLabel (kmerStr, cpt))));
      sc.trans.push_back (MachineTransition (string(), emitLabel(kmerStr), kmerEmit(kmer,cpt), WeightExpr (cptExtendLabel (kmerStr, cpt))));
      sc.trans.push_back (MachineTransition (string(), string(), kmerEnd(kmer), WeightExpr (cptEndLabel (kmerStr, cpt))));
    }
    for (auto c: alph)
      end.trans.push_back (MachineTransition (string(1,c), string(), kmerStart(stringToKmer(suffix+c,alph)), WeightExpr (true)));
    end.trans.push_back (MachineTransition (string(), string(), nStates() - 1, WeightExpr (padEndLabel())));
  }
  state[startState()].trans.push_back (MachineTransition (string(), padEmitLabel(), 0, padExtendLabel()));
  state[endState()].trans.push_back (MachineTransition (string(), padEmitLabel(), nStates() - 1, padExtendLabel()));
}
