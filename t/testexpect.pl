#!/usr/bin/env perl

use warnings;
use File::Temp;

die "Usage: $0 <prog> <args...> <expected>" unless @ARGV >= 2;
my $expected = pop @ARGV;
my ($prog, @args) = @ARGV;

my $fh = File::Temp->new();
my $fname = $fh->filename;

system "$prog @args >$fname";

if (-e $expected) {
    my $diff = `diff $fname $expected`;

    if (length $diff) {
	print "`$prog @args` does not match $expected:\n";
	print `diff -y $fname $expected`;
	die "not ok: `$prog @args`\n";
    } else {
	print "ok: `$prog @args` matches $expected\n";
    }

} else {

    my $actual = `cat $fname`;
    chomp $actual;
    
    if ($actual eq $expected) {
	print "ok: `$prog @args` = '$expected'\n";
    } else {
	print "`$prog @args` does not match '$expected':\n";
	print $actual, "\n";
	print "Possibly '$expected' is a filename? (in which case, file not found)\n";
	die "not ok: `$prog @args`\n";
    }
}
