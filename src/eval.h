#ifndef EVAL_INCLUDED
#define EVAL_INCLUDED

#include <algorithm>
#include "machine.h"
#include "params.h"

namespace MachineBoss {

template<typename Symbol,typename Token>
struct Tokenizer {
  vguard<Symbol> tok2sym;
  map<Symbol,Token> sym2tok;
  Tokenizer() { }
  Tokenizer (const vguard<Symbol>& symbols) {
    tok2sym.push_back (string());   // token zero is the empty string
    tok2sym.insert (tok2sym.end(), symbols.begin(), symbols.end());
    for (Token tok = 0; tok < (Token) tok2sym.size(); ++tok)
      sym2tok[tok2sym[tok]] = tok;
  }
  static inline Token emptyToken() { return 0; }
  vguard<Token> tokenize (const vguard<Symbol>& symSeq) const {
    vguard<Token> tokSeq;
    tokSeq.reserve (symSeq.size());
    for (const auto& sym: symSeq)
      tokSeq.push_back (sym2tok.at(sym));
    return tokSeq;
  }
  vguard<Symbol> detokenize (const vguard<Token>& tokSeq) const {
    vguard<Symbol> symSeq;
    symSeq.reserve (tokSeq.size());
    for (auto tok: tokSeq)
      symSeq.push_back (tok2sym[tok]);
    return symSeq;
  }
};

typedef int InputToken;
typedef int OutputToken;

typedef Tokenizer<InputSymbol,InputToken> InputTokenizer;
typedef Tokenizer<OutputSymbol,OutputToken> OutputTokenizer;

typedef double LogWeight;

struct EvaluatedMachineState {
  typedef size_t TransIndex;

  struct Trans {
    LogWeight logWeight;
    TransIndex transIndex;  // index of this transition in source state's TransList. Need to track this so we can map forward-backward counts back to MachineTransitions
  };
  typedef multimap<StateIndex,Trans> StateTransMap;
  typedef map<OutputToken,StateTransMap> OutStateTransMap;
  typedef map<InputToken,OutStateTransMap> InOutStateTransMap;
  
  StateName name;
  TransIndex nTransitions, transOffset;
  InOutStateTransMap incoming, outgoing;  // indexed by input token, output token, and (source or destination) state
  vguard<LogWeight> logTransWeight;  // indexed by TransIndex

  InputToken bestOutgoingToken (StateIndex dest, OutputToken out) const;  // for a given destination state & output token, find the best input token
};

struct EvaluatedMachine {
  InputTokenizer inputTokenizer;
  OutputTokenizer outputTokenizer;
  vguard<EvaluatedMachineState> state;
  EvaluatedMachineState::TransIndex nTransitions;
  EvaluatedMachine() { }
  EvaluatedMachine (const Machine&, const Params&);  // use machine.getParamDefs(true) to set missing parameters automatically
  EvaluatedMachine (const Machine&);  // WARNING: if this constructor is used, and no Params are supplied, all logWeight's will be zero
  void init (const Machine&, const Params*);
  void writeJson (ostream&) const;
  string toJsonString() const;
  StateIndex nStates() const;
  StateIndex startState() const;
  StateIndex endState() const;
  string stateNameJson (StateIndex) const;
  vguard<vguard<double> > sumInTrans (bool allPaths = false) const;  // returns effective transitions between states, summing over all non-outputting paths (or over ALL paths, if allPaths is true)
  vguard<vguard<LogWeight> > logSumInTrans (bool allPaths = false) const;  // log of sumInTrans(allPaths)
  Machine explicitMachine() const;  // returns the Machine without parameters, i.e. all transitions have numeric weights
  static vguard<InputSymbol> decode (const MachinePath&, const Machine&, const Params&);  // returns the input symbols for the most likely transition path consistent with the state path & output sequence specified by the MachinePath
};

}  // end namespace

#endif /* EVAL_INCLUDED */
