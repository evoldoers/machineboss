#ifndef BASECALL_INCLUDED
#define BASECALL_INCLUDED

#include "../machine.h"
#include "../fastseq.h"
#include "prior.h"

// Required for sane hyperparameters (see prior.h):
//  sigma_mean < sigma_sd

#define BaseCallerMuMean     100
#define BaseCallerMuError    20
#define BaseCallerSigmaMean  1
#define BaseCallerSigmaError 2

#define BaseCallerPadMuMean     BaseCallerMuMean
#define BaseCallerPadMuError    100
#define BaseCallerPadSigmaMean  50
#define BaseCallerPadSigmaError 60

struct BaseCallingParamNamer : EventFuncNamer {
  static string padEmitLabel();
  static string padExtendLabel();
  static string padEndLabel();
  static string emitLabel (const string& kmerStr);
  static string cptWeightLabel (const string& kmerStr, int cpt);
  static string kmerExtendLabel (const string& kmerStr);
  static string kmerEndLabel (const string& kmerStr);
  static string kmerExitRateLabel (const string& kmerStr);
  static string cptName (int cpt);
};

struct BaseCallingParams : BaseCallingParamNamer {
  string alphabet;
  SeqIdx kmerLen;
  int components;
  GaussianModelParams params;
  void init (const string& alph, SeqIdx kmerLen, int components);
  json asJson() const;
  void writeJson (ostream& out) const;
  void readJson (const json& json);
};

struct BaseCallingPrior : BaseCallingParamNamer, TraceParamsPrior {
  double cptWeight, padExtend, padEnd, kmerExitCount, kmerExitTime;
  double mu, muCount, tau, tauCount;
  double muPad, muPadCount, tauPad, tauPadCount;
  
  BaseCallingPrior();
  
  GaussianModelPrior modelPrior (const string& alph, SeqIdx kmerLen, int components) const;
};

struct BaseCallingMachine : Machine, BaseCallingParamNamer {
  int components, alphSize, nKmers, kmerOffset;
  void init (const string& alph, SeqIdx kmerLen, int components);
  // State indices are organized so that the only backward transitions (i->j where j<i) are output emissions
  inline StateIndex nShorterKmers (int len) const { return (numberOfKmers(len,alphSize) - 1) / (alphSize - 1); }  // sum_{n=0}^{L-1} A^l = (A^L - 1) / (A - 1)
  inline StateIndex shortKmer (Kmer kmer, int len) const { return len ? (nShorterKmers(len) + kmer) : 0; }
  inline StateIndex kmerEmit (Kmer kmer, int component) const { return kmerOffset + component * nKmers + kmer; }
  inline StateIndex kmerStart (Kmer kmer) const { return kmerOffset + components * nKmers + kmer; }
};

#endif /* BASECALL_INCLUDED */
