#!/usr/bin/env perl

use warnings;
use File::Temp;

die "Usage: $0 <testName> <testNameWidth> <prog> <args...> (<expected>|-fail|-idem)" unless @ARGV >= 3;
my $expected = pop @ARGV;
my ($test, $testWidth, @args) = @ARGV;

my $indentTest = sprintf ("% ${testWidth}s", $test);

my $fh = File::Temp->new();
my $fname = $fh->filename;

my $fhErr = File::Temp->new();
my $fnameErr = $fhErr->filename;

my $status = system "@args >$fname 2>$fnameErr";

my $idem;
if ($expected eq '-idem') {
    $expected = $args[$#args];
    $idem = 1;
}

if ($expected eq '-fail') {
    if ($status) {
	print "$indentTest     ok: `@args` failed on cue\n";
    } else {
	print "Standard output:\n", `cat $fname`;
	print "Standard error:\n", `cat $fnameErr`;
	die "$indentTest not ok: `@args` succeeded (was expected to fail)\n";
    }
} else {
    if ($status) {
	print "Standard output:\n", `cat $fname`;
	print "Standard error:\n", `cat $fnameErr`;
	die "$indentTest not ok: `@args` failed with exit code $status: $!\n";
    }

    if (-e $expected) {
	my $diff = `diff $fname $expected`;

	if (length $diff) {
	    print "`@args` does not match '$expected':\n";
	    print `diff -y $fname $expected`;
	    die "$indentTest not ok: `@args`\n";
	} else {
	    print "$indentTest     ok: `@args` ", ($idem ? "is idempotent" : "matches '$expected'"), "\n";
	}

    } else {
	my $actual = `cat $fname`;
	chomp $actual;
	
	if ($actual eq $expected) {
	    print "$indentTest     ok: `@args` = '$expected'\n";
	} else {
	    print "`@args` does not match '$expected':\n";
	    print $actual, "\n";
	    print "Possibly '$expected' is a filename? (in which case, file not found)\n";
	    die "$indentTest not ok: `@args`\n";
	}
    }
}
