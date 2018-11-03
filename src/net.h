#ifndef NET_INCLUDED
#define NET_INCLUDED

#include "fastseq.h"
#include "hmmer.h"

FastSeq getUniprot (const string& id);
HmmerModel getPfam (const string& id);
HmmerModel getDfam (const string& id);

string inflateString (const string&);

#endif /* NET_INCLUDED */
