#include <iostream>
#include <string>
#include <zlib.h>
#include "net.h"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "../../ext/cpp-httplib/httplib.h"

using namespace std;
using namespace MachineBoss;

const char* host = "dfam.org";
const int port = 80;

const char* prefix = "/download/model/";
const char* suffix = "?file=hmm";

int main (int argc, char** argv)
{
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " <PFAM_ID>" << endl;
    exit(1);
  }

  httplib::Client cli (host, port);
  //  httplib::Params params;
  //  params.emplace("file","hmm");
  
  const string path = string(prefix) + argv[1] + suffix;
  cerr << "Path: " << path << endl;
  auto res = cli.Get (path.c_str());

  if (res) {
    cerr << "Status: " << res->status << endl;
    if (res->has_header("Location"))
      cerr << "Location: " << res->get_header_value("Location") << endl;

    const string decompressed = inflateString (res->body);
    cout << decompressed << endl;
  }
}
