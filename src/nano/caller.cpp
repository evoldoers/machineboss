#include "caller.h"
#include "../logger.h"

#define BaseCallerMuMode BaseCallerMuMean
#define BaseCallerPadMuMode BaseCallerPadMuMean
#define BaseCallerTauMode CALC_TAU_MODE (CALC_TAU_MEAN(BaseCallerSigmaMean), CALC_TAU_SD(BaseCallerSigmaError))
#define BaseCallerPadTauMode CALC_TAU_MODE (CALC_TAU_MEAN(BaseCallerPadSigmaMean), CALC_TAU_SD(BaseCallerPadSigmaError))

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

string BaseCallingParamNamer::kmerExtendLabel (const string& kmerStr) {
  return waitEventFuncName (kmerExitRateLabel (kmerStr));
}

string BaseCallingParamNamer::kmerEndLabel (const string& kmerStr) {
  return exitEventFuncName (kmerExitRateLabel (kmerStr));
}

string BaseCallingParamNamer::kmerExitRateLabel (const string& kmerStr) {
  return string("R(") + kmerStr + ")";
}

string BaseCallingParamNamer::cptName (int cpt) {
  return string("k=") + to_string(cpt + 1);
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
    if (cpts > 1)
      for (int cpt = 0; cpt < cpts; ++cpt)
	params.prob.defs[cptWeightLabel (kmerStr, cpt)] = 1. / (double) cpts;
    params.rate.defs[kmerExitRateLabel (kmerStr)] = 1.;
    GaussianParams gp;
    gp.mu = BaseCallerMuMode;
    gp.tau = BaseCallerTauMode;
    params.gauss[emitLabel (kmerStr)] = gp;
  }
  GaussianParams pad_gp;
  pad_gp.mu = BaseCallerPadMuMode;
  pad_gp.tau = BaseCallerPadTauMode;
  params.gauss[padEmitLabel()] = pad_gp;
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
    kmerExitCount(1),
    kmerExitTime(1),
    mu (BaseCallerMuMode),
    muCount (CALC_N_MU (BaseCallerMuError, CALC_TAU_MEAN(BaseCallerSigmaMean), CALC_TAU_SD(BaseCallerSigmaError))),
    tau (BaseCallerTauMode),
    tauCount (CALC_N_TAU (CALC_TAU_MEAN(BaseCallerSigmaMean), CALC_TAU_SD(BaseCallerSigmaError))),
    muPad (BaseCallerPadMuMode),
    muPadCount (CALC_N_MU (BaseCallerPadMuError, CALC_TAU_MEAN(BaseCallerPadSigmaMean), CALC_TAU_SD(BaseCallerPadSigmaError))),
    tauPad (BaseCallerPadTauMode),
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
  exitPrior.count = kmerExitCount;
  exitPrior.time = kmerExitTime;
  
  const Kmer nk = numberOfKmers (kmerLen, alph.size());
  for (Kmer kmerPrefix = 0; kmerPrefix < nk; kmerPrefix += alph.size())
    for (Kmer kmer = kmerPrefix; kmer < kmerPrefix + alph.size(); ++kmer) {
      const string kmerStr = kmerToString (kmer, kmerLen, alph);
      const string prefix = kmerStr.substr(0,kmerLen-1);
      const char suffix = kmerStr[kmerLen-1];
      if (components > 1) {
	vguard<string> cptWeightParam;
	cptWeightParam.reserve (components);
	for (int cpt = 0; cpt < components; ++cpt) {
	  prior.count.defs[cptWeightLabel (kmerStr, cpt)] = cptWeight;
	  cptWeightParam.push_back (cptWeightLabel (kmerStr, cpt));
	}
	prior.cons.norm.push_back (cptWeightParam);
      }
      prior.gamma[kmerExitRateLabel (kmerStr)] = exitPrior;
      prior.cons.rate.push_back (kmerExitRateLabel (kmerStr));
      prior.gauss[emitLabel (kmerStr)] = emitPrior;
    }

  GaussianPrior padPrior;
  padPrior.mu0 = muPad;
  padPrior.n_mu = muPadCount;
  padPrior.tau0 = tauPad;
  padPrior.n_tau = tauPadCount;

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
  state = vguard<MachineState> (nKmers * (components + 1) + kmerOffset + 1);
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
    const WeightExpr endWeight (kmerEndLabel (kmerStr)), extendWeight (kmerExtendLabel (kmerStr));
    MachineState& start (state[kmerStart(kmer)]);
    start.name = kmerStr + "_start";
    for (int cpt = 0; cpt < cpts; ++cpt) {
      MachineState& sc = state[kmerEmit(kmer,cpt)];
      sc.name = kmerStr + "(" + cptName(cpt) + ")";
      start.trans.push_back (MachineTransition (string(), emitLabel(kmerStr), kmerEmit(kmer,cpt), cpts == 1 ? WeightExpr(true) : WeightExpr (cptWeightLabel (kmerStr, cpt))));
      sc.trans.push_back (MachineTransition (string(), emitLabel(kmerStr), kmerEmit(kmer,cpt), extendWeight));
      if (cpt + 1 < cpts)
	sc.trans.push_back (MachineTransition (string(), string(), kmerEmit(kmer,cpt+1), endWeight));
    }
    MachineState& end = state[kmerEmit(kmer,cpts-1)];
    for (auto c: alph)
      end.trans.push_back (MachineTransition (string(1,c), string(), kmerStart(stringToKmer(suffix+c,alph)), endWeight));
    end.trans.push_back (MachineTransition (string(), string(), nStates() - 1, WeightAlgebra::multiply (endWeight, WeightExpr (padEndLabel()))));
  }
  state[startState()].trans.push_back (MachineTransition (string(), padEmitLabel(), 0, padExtendLabel()));
  state[endState()].trans.push_back (MachineTransition (string(), padEmitLabel(), nStates() - 1, padExtendLabel()));
}
