#ifndef SCORES_INCLUDED
#define SCORES_INCLUDED

#include "trans.h"
#include "params.h"

typedef int Token;
typedef Token InputToken;
typedef Token OutputToken;
typedef pair<InputToken,OutputToken> MachineTransitionLabel;

template<class AlphabetSymbol,class AlphabetToken>
struct Tokenizer {
  vguard<AlphabetSymbol> tok2sym;
  map<AlphabetSymbol,AlphabetToken> sym2tok;
  Tokenizer (const vguard<AlphabetSymbol>& symbols) {
    tok2sym.push_back (string());   // token zero is the empty string
    tok2sym.insert (tok2sym.end(), symbols.begin(), symbols.end());
    for (AlphabetToken tok = 0; tok < tok2sym.size(); ++tok)
      sym2tok[tok2sym[tok]] = tok;
  }
  inline AlphabetToken emptyToken() const { return 0; }
  inline AlphabetToken badToken() const { return -1; }
  vguard<AlphabetToken> tokenize (const vguard<AlphabetSymbol>& symSeq) const {
    vguard<AlphabetToken> tokSeq;
    tokSeq.reserve (symSeq.size());
    for (const auto& sym: symSeq)
      tokSeq.push_back (sym2tok.count(sym) ? sym2tok.at(sym) : badToken());
    return tokSeq;
  }
};
typedef Tokenizer<InputSymbol,InputToken> InputTokenizer;
typedef Tokenizer<OutputSymbol,OutputToken> OutputTokenizer;

struct MachineTransitionScore {
  InputSymbol in;
  OutputSymbol out;
  StateIndex src, dest;
  double logWeight;

  MachineTransitionScore (StateIndex src, const MachineTransition&, const Params&, const InputTokenizer&, const OutputTokenizer&);
  MachineTransitionLabel label() const;
};

struct MachineStateScores {
  map<MachineTransitionLabel,MachineTransitionScore> incoming, outgoing;
};

struct MachineScores {
  InputTokenizer inputTokenizer;
  OutputTokenizer outputTokenizer;
  vguard<MachineStateScores> state;
  MachineStateScores (const Machine&, const Params&);
};

#endif /* SCORES_INCLUDED */
