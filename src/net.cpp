#include "net.h"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "../../ext/cpp-httplib/httplib.h"

FastSeq getUniprot (const string& id) {
  const char* host = "www.uniprot.org";
  const int port = 443;
  const char* prefix = "/uniprot/";
  const char* suffix = ".fasta";

  httplib::SSLClient cli (host, port);
  const string path = string(prefix) + id + suffix;
  auto res = cli.Get (path.c_str());
  if (res->status != -1 && res->status != 200)
    return FastSeq();
  return FastSeq::fromFasta (string (res->body));
}

HmmerModel getPfam (const string& id) {
  HmmerModel hmm;

  const char* host = "pfam.xfam.org";
  const int port = 80;
  const char* prefix = "/family/";
  const char* suffix = "/hmm";

  httplib::Client cli (host, port);
  const string path = string(prefix) + id + suffix;
  auto res = cli.Get (path.c_str());

  istringstream iss (string (res->body));
  hmm.read (iss);
  return hmm;
}

HmmerModel getDfam (const string& id) {
  HmmerModel hmm;

  const char* host = "dfam.org";
  const int port = 80;
  const char* prefix = "/download/model/";
  const char* suffix = "?file=hmm";

  httplib::Client cli (host, port);
  const string path = string(prefix) + id + suffix;
  auto res = cli.Get (path.c_str());

  istringstream iss (inflateString (res->body));
  hmm.read (iss);
  return hmm;
}

string inflateString (const string& compressed) {
  size_t bufSize = compressed.size() * 2;
  char* buf = new char[bufSize];
  string decompressed;
    
  z_stream infstream;
  infstream.zalloc = Z_NULL;
  infstream.zfree = Z_NULL;
  infstream.opaque = Z_NULL;
  infstream.avail_in = (uInt)(compressed.size()); // size of input
  infstream.next_in = (Bytef *)(compressed.c_str()); // input char array
     
  inflateInit2(&infstream, 16+MAX_WBITS);
  int retcode, last_out = 0;
  do {
    infstream.avail_out = (uInt)bufSize; // size of output
    infstream.next_out = (Bytef *)buf; // output char array
    retcode = inflate(&infstream, Z_NO_FLUSH);
    Assert (retcode == Z_OK || retcode == Z_STREAM_END, "Error during decompression");
    decompressed.insert (decompressed.end(), buf, buf + infstream.total_out - last_out);
    last_out = infstream.total_out;
  } while (retcode != Z_STREAM_END);
  inflateEnd(&infstream);

  delete[] buf;
  
  return decompressed;
}
