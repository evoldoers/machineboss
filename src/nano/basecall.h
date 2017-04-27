#ifndef BASECALL_INCLUDED
#define BASECALL_INCLUDED

#include "../machine.h"
#include "gaussian.h"

struct BaseCallingParamNamer {
  static string emitLabel (const string& kmerStr);
  static string condFreqLabel (const string& prefix, const char suffix);
  static string cptWeightLabel (const string& kmerStr, int cpt);
  static string cptExtendLabel (const string& kmerStr, int cpt);
  static string cptEndLabel (const string& kmerStr, int cpt);
  static string cptName (int cpt);
};

struct BaseCallingParams : BaseCallingParamNamer {
  string alphabet;
  SeqIndex kmerLen;
  int components;
  GaussianModelParams params;
  void init (const string& alph, SeqIndex kmerLen, int components);
  json asJson() const;
  void writeJson (ostream& out) const;
  void readJson (const json& json);
};

struct BaseCallingPrior : BaseCallingParamNamer {
  double condFreq, cptWeight, cptExtend, cptEnd;
  double mu, muCount;
  double tau, tauCount;

  BaseCallingPrior();
  
  GaussianModelPrior modelPrior (const string& alph, SeqIndex kmerLen, int components) const;
};

struct BaseCallingMachine : Machine, BaseCallingParamNamer {
  inline StateIndex kmerStart (Kmer kmer) const { return kmer * (components + 2) + 1; }
  void init (const string& alph, SeqIndex kmerLen, int components);
};

#endif /* BASECALL_INCLUDED */
