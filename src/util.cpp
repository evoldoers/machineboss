// feature test macro requirement for ftw
// #define _XOPEN_SOURCE 500

// includes
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <ftw.h>

#include "util.h"
#include "stacktrace.h"
#include "logger.h"

// buffer size for popen
#define PIPE_BUF_SIZE 1024

// buffer size for getcwd
#define DIR_BUF_SIZE 4096

// function defs
void Warn(const char* warning, ...) {
  va_list argptr;
  fprintf(stderr,"Warning: ");
  va_start (argptr, warning);
  vfprintf(stderr,warning,argptr);
  fprintf(stderr,"\n");
  va_end (argptr);
}

void Abort(const char* error, ...) {
  va_list argptr;
  va_start (argptr, error);
  fprintf(stderr,"Abort: ");
  vfprintf(stderr,error,argptr);
  fprintf(stderr,"\n");
  va_end (argptr);
  printStackTrace();
  throw;
}

void Fail(const char* error, ...) {
  va_list argptr;
  va_start (argptr, error);
  vfprintf(stderr,error,argptr);
  fprintf(stderr,"\n");
  va_end (argptr);
  exit (EXIT_FAILURE);
}

std::string plural (long n, const char* singular) {
  std::string s = std::to_string(n) + " " + singular;
  if (n != 1)
    s += "s";
  return s;
}

std::string plural (long n, const char* singular, const char* plural) {
  std::string s = std::to_string(n) + " " + (n == 1 ? singular : plural);
  return s;
}

std::vector<std::string> split (const std::string& s, const char* splitChars) {
  std::vector<std::string> result;
  auto b = s.begin();
  while (true) {
    while (b != s.end() && strchr (splitChars, *b) != NULL)
      ++b;
    if (b == s.end())
      break;
    auto e = b;
    while (e != s.end() && strchr (splitChars, *e) == NULL)
      ++e;
    result.push_back (string (b, e));
    b = e;
  }
  return result;
}

std::string toupper (const std::string& s) {
  std::string r (s);
  for (size_t n = 0; n < r.size(); ++n)
    r[n] = toupper (r[n]);
  return r;
}
