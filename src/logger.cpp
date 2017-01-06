#include <sstream>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include "logger.h"
#include "regexmacros.h"

Logger logger;

// POSIX basic regular expressions
const regex all_v ("-" RE_PLUS("v"), regex_constants::basic);
const regex numeric_v ("-v" RE_NUMERIC_GROUP, regex_constants::basic);

// functions
string ansiEscape (int code) {
  return string("\x1b[") + to_string(code) + "m";
}

Logger::Logger()
  : verbosity(0), useAnsiColor(true)
{
  for (int col : { 7, 2, 3, 5, 6, 1, 2, 3, 5, 6 })  // no blue, it's invisible
    logAnsiColor.push_back (ansiEscape(30 + col) + ansiEscape(40));
  threadAnsiColor = ansiEscape(37) + ansiEscape(41);  // white on red
  ansiColorOff = ansiEscape(0);

  setThreadName (this_thread::get_id(), "main thread");
}

void Logger::addTag (const char* tag) {
  addTag (string (tag));
}

void Logger::addTag (const string& tag) {
  logTags.insert (tag);
}

void Logger::setVerbose (int v) {
  verbosity = max (verbosity, v);
}

void Logger::colorOff() {
  useAnsiColor = false;
}

void Logger::parseLogArgs (boost::program_options::variables_map& vm) {
  setVerbose (vm.at("verbose").as<int>());
  if (vm.count("log"))
    for (const auto& x: vm.at("log").as<vector<string> >())
      addTag (x);

  if (vm.count("nocolor"))
    useAnsiColor = false;
}

void Logger::lock (int color, const char* file, int line, bool banner) {
  thread::id myId = this_thread::get_id();
  if (mx.try_lock_for (std::chrono::milliseconds(1000))) {
    if (lastMxOwner != myId && banner && threadName.size() > 1)
      clog << (useAnsiColor ? threadAnsiColor.c_str() : "")
	   << '(' << getThreadName(myId) << ')'
	   << (useAnsiColor ? ansiColorOff.c_str() : "") << ' ';
    lastMxOwner = myId;
    mxOwnerFile = file;
    mxOwnerLine = line;
  } else if (banner)
    clog << (useAnsiColor ? threadAnsiColor.c_str() : "")
	 << '(' << getThreadName(myId) << ", ignoring lock by " << getThreadName(lastMxOwner) << " at " << mxOwnerFile << " line " << mxOwnerLine << ')'
	 << (useAnsiColor ? ansiColorOff.c_str() : "") << ' ';
  if (banner && useAnsiColor)
    clog << (color < 0
	     ? logAnsiColor.front()
	     : (color >= (int) logAnsiColor.size()
		? logAnsiColor.back()
		: logAnsiColor[color]));
}

void Logger::unlock (bool banner) {
  if (banner && useAnsiColor)
    clog << ansiColorOff;
  mx.unlock();
}

string Logger::getThreadName (thread::id id) {
  const auto& iter = threadName.find(id);
  if (iter == threadName.end()) {
    ostringstream o;
    o << "thread " << id;
    return o.str();
  }
  return iter->second;
}

void Logger::setThreadName (thread::id id, const string& name) {
  threadName[id] = name;
}

void Logger::nameLastThread (const list<thread>& threads, const char* prefix) {
  setThreadName (threads.back().get_id(),
		 string(prefix) + " thread #" + to_string(threads.size()));
}

void Logger::eraseThreadName (const thread& thr) {
  threadName.erase (thr.get_id());
}

ProgressLogger::ProgressLogger (int verbosity, const char* function, const char* file, int line)
  : msg(NULL), verbosity(verbosity), function(function), file(file), line(line)
{ }

void ProgressLogger::initProgress (const char* desc, ...) {
  startTime = std::chrono::system_clock::now();
  lastElapsedSeconds = 0;
  reportInterval = 2;

  time_t rawtime;
  struct tm * timeinfo;

  time (&rawtime);
  timeinfo = localtime (&rawtime);
  
  va_list argptr;
  va_start (argptr, desc);
  vasprintf (&msg, desc, argptr);
  va_end (argptr);

  if (logger.testVerbosityOrLogTags (verbosity, function, file)) {
    ostringstream l;
    l << msg << ": started at " << asctime(timeinfo);
    logger.print (l.str(), file, line, verbosity);
  }
}

ProgressLogger::~ProgressLogger() {
  if (msg)
    free (msg);
}

void ProgressLogger::logProgress (double completedFraction, const char* desc, ...) {
  va_list argptr;
  const std::chrono::system_clock::time_point currentTime = std::chrono::system_clock::now();
  const auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds> (currentTime - startTime).count();
  const double estimatedTotalSeconds = elapsedSeconds / completedFraction;
  if (elapsedSeconds > lastElapsedSeconds + reportInterval) {
    const double estimatedSecondsLeft = estimatedTotalSeconds - elapsedSeconds;
    const double estimatedMinutesLeft = estimatedSecondsLeft / 60;
    const double estimatedHoursLeft = estimatedMinutesLeft / 60;
    const double estimatedDaysLeft = estimatedHoursLeft / 24;

    if (completedFraction > 0 && logger.testVerbosityOrLogTags (verbosity, function, file)) {
      char *progMsg;
      va_start (argptr, desc);
      vasprintf (&progMsg, desc, argptr);
      va_end (argptr);

      ostringstream l;
      l << msg << ": " << progMsg << ". Estimated time left: ";
      if (estimatedDaysLeft > 2)
	l << estimatedDaysLeft << " days";
      else if (estimatedHoursLeft > 2)
	l << estimatedHoursLeft << " hrs";
      else if (estimatedMinutesLeft > 2)
	l << estimatedMinutesLeft << " mins";
      else
	l << estimatedSecondsLeft << " secs";
      l << " (" << (100*completedFraction) << "%)" << endl;

      logger.print (l.str(), file, line, verbosity);
      
      free(progMsg);
    }
    
    lastElapsedSeconds = elapsedSeconds;
    reportInterval = fmin (10., 2*reportInterval);
  }
}
