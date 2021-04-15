#include <iostream>
#include "fastseq.h"
#include "util.h"
#include "logger.h"
#include "regexmacros.h"

KSEQ_INIT(gzFile, gzread)

const regex fasta_re (">([^ ]+)(.*)");
using namespace MachineBoss;

FastSeq FastSeq::fromFasta (const string& fasta) {
  FastSeq fs;  
  smatch match;
  const vguard<string> fastaLine = split (fasta, "\n");
  if (fastaLine.size() && regex_match (fastaLine[0], match, fasta_re)) {
    fs.name = match.str (1);
    if (match.str(2).size() > 1)
      fs.comment = match.str(2).substr(1);
    for (size_t n = 1; n < fastaLine.size(); ++n) {
      if (regex_search (fastaLine[n], match, fasta_re))
	Abort ("Multiple sequences in FASTA string");
      for (const auto& nonwhite: split (fastaLine[n]))
	fs.seq += nonwhite;
    }
  } else
    Warn ("Does not look like FASTA\n%s\n", fasta.c_str());
  return fs;
}

FastSeq FastSeq::fromSeq (const string& seq, const string& name) {
  FastSeq fs;
  fs.name = name;
  fs.seq = seq;
  return fs;
}

UnvalidatedAlphTok MachineBoss::tokenize (char c, const string& alphabet) {
  const char* alphStr = alphabet.c_str();
  const char* ptok = strchr (alphStr, c);
  if (ptok == NULL)
    ptok = strchr (alphStr, std::isupper(c) ? std::tolower(c) : std::toupper(c));
  return ptok ? (UnvalidatedAlphTok) (ptok - alphStr) : -1;
}

bool MachineBoss::isValidToken (char c, const string& alphabet) {
  return tokenize(c,alphabet) >= 0;
}

const char FastSeq::minQualityChar = '!';
const char FastSeq::maxQualityChar = '~';
const QualScore FastSeq::qualScoreRange = 94;

TokSeq MachineBoss::validTokenize (const string& s, const string& alphabet, const char* seqname) {
  TokSeq tok;
  tok.reserve (s.size());
  for (auto c: s) {
    const UnvalidatedAlphTok t = tokenize (c, alphabet);
    if (t < 0) {
      cerr << "Unknown symbol " << c << " in sequence"
	   << (seqname ? ((string(" ") + seqname)) : string())
	   << " (alphabet is " << alphabet << ")" << endl;
      throw;
    }
    tok.push_back ((AlphTok) t);
  }
  return tok;
}

string MachineBoss::detokenize (const TokSeq& toks, const string& alphabet) {
  string seq;
  seq.reserve (toks.size());
  for (auto tok : toks) {
    if (tok >= alphabet.size()) {
      cerr << "Unknown token " << tok << " in sequence (alphabet is " << alphabet << ")" << endl;
      throw;
    }
    seq.push_back (alphabet[tok]);
  }
  return seq;
}

TokSeq FastSeq::tokens (const string& alphabet) const {
  return validTokenize (seq, alphabet, name.c_str());
}

Kmer MachineBoss::makeKmer (SeqIdx k, vector<unsigned int>::const_iterator tok, AlphTok alphabetSize) {
  Kmer kmer = 0, mul = 1;
  for (SeqIdx j = 0; j < k; ++j) {
    const unsigned int token = tok[k - j - 1];
    kmer += mul * token;
    mul *= alphabetSize;
  }
  return kmer;
}

Kmer MachineBoss::numberOfKmers (SeqIdx k, AlphTok alphabetSize) {
  Kmer n;
  for (n = 1; k > 0; --k)
    n *= alphabetSize;
  return n;
}

string MachineBoss::kmerToString (Kmer kmer, SeqIdx k, const string& alphabet) {
  string rev;
  for (SeqIdx j = 0; j < k; ++j, kmer = kmer / alphabet.size())
    rev += alphabet[kmer % alphabet.size()];
  return string (rev.rbegin(), rev.rend());
}

Kmer MachineBoss::stringToKmer (const string& s, const string& alphabet) {
  const TokSeq tok = validTokenize (s, alphabet);
  return makeKmer (tok.size(), tok.begin(), alphabet.size());
}

void FastSeq::writeFasta (ostream& out) const {
  out << '>' << name;
  if (comment.size())
    out << ' ' << comment;
  out << endl;
  const size_t width = DefaultFastaCharsPerLine;
  for (size_t i = 0; i < seq.size(); i += width)
    out << seq.substr(i,width) << endl;
}

void FastSeq::writeFastq (ostream& out) const {
  out << '@' << name;
  if (comment.size())
    out << ' ' << comment;
  out << endl;
  out << seq << endl;
  if (hasQual())
    out << '+' << endl << qual << endl;
}

void MachineBoss::writeFastaSeqs (ostream& out, const vguard<FastSeq>& fastSeqs) {
  for (const auto& s : fastSeqs)
    s.writeFasta (out);
}

void MachineBoss::writeFastqSeqs (ostream& out, const vguard<FastSeq>& fastSeqs) {
  for (const auto& s : fastSeqs)
    s.writeFastq (out);
}

void initFastSeq (FastSeq& seq, kseq_t* ks) {
  if (ks->name.l)
    seq.name = string(ks->name.s);
  if (ks->seq.l)
    seq.seq = string(ks->seq.s);
  if (ks->comment.l)
    seq.comment = string(ks->comment.s);
  if (ks->qual.l && ks->qual.l == ks->seq.l)
    seq.qual = string(ks->qual.s);
}

void MachineBoss::readFastSeqs (const char* filename, vguard<FastSeq>& seqs) {
  gzFile fp = gzopen(filename, "r");
  Require (fp != Z_NULL, "Couldn't open %s", filename);

  kseq_t *ks = kseq_init(fp);
  while (true) {
    if (kseq_read(ks) == -1)
      break;

    FastSeq seq;
    initFastSeq (seq, ks);

    seqs.push_back (seq);
  }
  kseq_destroy (ks);
  gzclose (fp);

  LogThisAt(3, "Read " << plural(seqs.size(),"sequence") << " from " << filename << endl);
  
  if (seqs.empty())
    Warn ("Couldn't read any sequences from %s", filename);
}

vguard<FastSeq> MachineBoss::readFastSeqs (const char* filename) {
  vguard<FastSeq> seqs;
  readFastSeqs (filename, seqs);
  return seqs;
}

set<string> MachineBoss::fastSeqDuplicateNames (const vguard<FastSeq>& seqs) {
  set<string> name, dups;
  for (const auto& s : seqs) {
    if (name.find(s.name) != name.end())
      dups.insert (s.name);
    name.insert (s.name);
  }
  return dups;
}

KmerIndex::KmerIndex (const FastSeq& seq, const string& alphabet, SeqIdx kmerLen)
  : seq(seq), alphabet(alphabet), kmerLen(kmerLen)
{
  LogThisAt(5, "Building " << kmerLen << "-mer index for " << seq.name << endl);
  const TokSeq tok = seq.tokens (alphabet);
  const AlphTok alphabetSize = (AlphTok) alphabet.size();
  const SeqIdx seqLen = seq.length();
  for (SeqIdx j = 0; j <= seqLen - kmerLen; ++j)
    kmerLocations[makeKmer (kmerLen, tok.begin() + j, alphabetSize)].push_back (j);

  if (LoggingThisAt(8)) {
    LogStream (8, "Frequencies of " << kmerLen << "-mers in " << seq.name << ':' << endl);
    for (const auto& kl : kmerLocations) {
      LogStream (8, kmerToString (kl.first, kmerLen, alphabet) << ' ' << kl.second.size() << endl);
    }
  }
}

