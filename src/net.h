#ifndef NET_INCLUDED
#define NET_INCLUDED

#include "fastseq.h"
#include "hmmer.h"

namespace MachineBoss {

FastSeq getUniprot (const string& id);
HmmerModel getPfam (const string& id);
HmmerModel getDfam (const string& id);

string inflateString (const string&);

}  // end namespace

#endif /* NET_INCLUDED */
