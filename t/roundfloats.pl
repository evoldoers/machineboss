#!/usr/bin/env perl

use warnings;
use IPC::Open3;

die "Usage: $0 <maxPrecision> <prog> <args...>" unless @ARGV >= 2;
my $precision = shift;
my ($prog, @args) = @ARGV;

my $pid = open3(\*CHILD_IN, \*CHILD_OUT, \*CHILD_ERR,
		$prog, @args);

close CHILD_IN;
my @out = <CHILD_OUT>;
my @err = <CHILD_ERR>;
waitpid ($pid, 0);

grep s/(\d+\.\d{$precision,})/@{[sprintf("%.${precision}g",$1)]}/g, @out;
print @out;

select (STDERR);
print @err;
