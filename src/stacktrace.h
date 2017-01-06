#ifndef STACKTRACE_INCLUDED
#define STACKTRACE_INCLUDED

#include <stdio.h>
#include <execinfo.h>

// http://oroboro.com/stack-trace-on-crash/
static inline void printStackTrace( FILE *out = stderr, unsigned int max_frames = 63 ) {
  fprintf(out, "Stack trace:\n");
 
  // storage array for stack trace address data
  void* addrlist[max_frames+1];
 
  // retrieve current stack addresses
  auto addrlen = backtrace( addrlist, ((int) sizeof( addrlist )) / (int) sizeof( void* ));
 
  if ( addrlen == 0 ) {
    fprintf( out, "  \n" );
    return;
  }
 
  // create readable strings to each frame.
  char** symbollist = backtrace_symbols( addrlist, addrlen );
 
  // print the stack trace.
  auto i = addrlen;
  for ( i = 1; i < addrlen; i++ )
    fprintf( out, "%s\n", symbollist[i] );
 
  free(symbollist);
}

#endif /* STACKTRACE_INCLUDED */
