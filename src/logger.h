#ifndef LOGGER_INCLUDED
#define LOGGER_INCLUDED

#include <list>
#include <set>
#include <map>
#include <string>
#include <deque>
#include <mutex>
#include <thread>
#include <ratio>
#include <chrono>
#include <iostream>
#include <sstream>
#include <boost/program_options.hpp>
#include "util.h"
#include "vguard.h"

using namespace std;

class Logger {
private:
  int verbosity;
  set<string> logTags;
  bool useAnsiColor;
  vguard<string> logAnsiColor;
  string threadAnsiColor, ansiColorOff;
  
  recursive_timed_mutex mx;
  thread::id lastMxOwner;
  const char* mxOwnerFile;
  int mxOwnerLine;
  map<thread::id,string> threadName;

public:
  Logger();
  // configuration
  void addTag (const char* tag);
  void addTag (const string& tag);
  void setVerbose (int v);
  void colorOff();
  void parseLogArgs (boost::program_options::variables_map& vm);
  
  inline bool testVerbosity (int v) {
    return verbosity >= v;
  }

  inline bool testLogTag (const char* tag) {
    return logTags.find(tag) != logTags.end();
  }

  inline bool testVerbosityOrLogTags (int v, const char* tag1, const char* tag2) {
    return verbosity >= v || testLogTag(tag1) || testLogTag(tag2);
  }

  string getThreadName (thread::id id);
  void setThreadName (thread::id id, const string& name);
  void nameLastThread (const list<thread>& threads, const char* prefix);
  void eraseThreadName (const thread& thr);

  void lock (int color = 0, const char* file = "", const int line = 0, bool banner = true);
  void unlock (bool endBanner = true);
  void lockSilently() { lock(0,"",0,false); }
  void unlockSilently() { unlock(false); }

  template<class T>
  void print (const T& t, const char* file, int line, int v) {
    lock(v,file,line,true);
    clog << t;
    unlock(true);
  }
};

extern Logger logger;

#define LoggingAt(V)     (logger.testVerbosity(V))
#define LoggingThisAt(V) (logger.testVerbosityOrLogTags(V,__func__,__FILE__))
#define LoggingTag(T)    (logger.testLogTag(T))

#define LogStream(V,S) do { ostringstream tmpLog; tmpLog << S; logger.print(tmpLog.str(),__FILE__,__LINE__,V); } while(0)

#define LogAt(V,S)     do { if (LoggingAt(V)) LogStream(V,S); } while(0)
#define LogThisAt(V,S) do { if (LoggingThisAt(V)) LogStream(V,S); } while(0)
#define LogThisIf(X,S) do { if (X) LogStream(0,S); } while(0)


/* progress logging */
class ProgressLogger {
public:
  std::chrono::system_clock::time_point startTime;
  double lastElapsedSeconds, reportInterval;
  char* msg;
  int verbosity;
  const char *function, *file;
  int line;
  ProgressLogger (int verbosity, const char* function, const char* file, int line);
  ~ProgressLogger();
  void initProgress (const char* desc, ...);
  void logProgress (double completedFraction, const char* desc, ...);
private:
  ProgressLogger (const ProgressLogger&) = delete;
  ProgressLogger& operator= (const ProgressLogger&) = delete;
};

#define ProgressLog(PLOG,V) ProgressLogger PLOG (V, __func__, __FILE__, __LINE__)

#endif /* LOGGER_INCLUDED */

