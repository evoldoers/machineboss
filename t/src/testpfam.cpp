#include <iostream>
#include <string>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "../../ext/cpp-httplib/httplib.h"

using namespace std;
using namespace MachineBoss;

const char* host = "pfam.xfam.org";
const int port = 80;

const char* prefix = "/family/";
const char* suffix = "/hmm";

int main (int argc, char** argv)
{
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " <PFAM_ID>" << endl;
    exit(1);
  }

  httplib::Client cli (host, port);
  
  const string path = string(prefix) + argv[1] + suffix;
  cerr << "Path: " << path << endl;
  auto res = cli.Get (path.c_str());

  if (res) {
    cerr << "Status: " << res->status << endl;
    if (res->has_header("Location"))
      cerr << "Location: " << res->get_header_value("Location") << endl;
    cout << res->body << endl;
  }
}
