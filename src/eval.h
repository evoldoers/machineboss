#ifndef EVAL_INCLUDED
#define EVAL_INCLUDED

#include <algorithm>
#include "machine.h"
#include "params.h"

template<typename Symbol,typename Token>
struct Tokenizer {
  vguard<Symbol> tok2sym;
  map<Symbol,Token> sym2tok;
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
    for (auto sym: symSeq)
      tokSeq.push_back (sym2tok.at(sym));
    return tokSeq;
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
    void init (LogWeight, TransIndex);
  };
  typedef map<StateIndex,Trans> StateTransMap;
  typedef map<OutputToken,StateTransMap> OutStateTransMap;
  typedef map<InputToken,OutStateTransMap> InOutStateTransMap;
  
  StateName name;
  TransIndex nTransitions;
  InOutStateTransMap incoming, outgoing;  // indexed by input token, output token, and (source or destination) state
};

struct EvaluatedMachine {
  InputTokenizer inputTokenizer;
  OutputTokenizer outputTokenizer;
  vguard<EvaluatedMachineState> state;
  EvaluatedMachine (const Machine&, const Params&);
  EvaluatedMachine (const Machine&);  // if no Params are supplied, all logWeight's will be zero
  void init (const Machine&, const Params*);
  void writeJson (ostream&) const;
  string toJsonString() const;
  StateIndex nStates() const;
  StateIndex startState() const;
  StateIndex endState() const;
  string stateNameJson (StateIndex) const;
};

#endif /* EVAL_INCLUDED */
