#!/usr/bin/env perl

use warnings;
use File::Temp;

die "Usage: $0 <testName> <testNameWidth> <prog> <args...> (<expected>|-fail|-idem)" unless @ARGV >= 3;
my $expected = pop @ARGV;
my ($test, $testWidth, @args) = @ARGV;

my $indentTest = sprintf ("% ${testWidth}s", $test);

my $fhOut = File::Temp->new();
my $fnameOut = $fhOut->filename;

my $fhErr = File::Temp->new();
my $fnameErr = $fhErr->filename;

my $status = system (join(" ",$args[0],map("'$_'", @args[1..$#args])) . " >$fnameOut 2>$fnameErr");

my $idem;
if ($expected eq '-idem') {
    $expected = $args[$#args];
    $idem = 1;
}

my ($reset, $bold, $red, $green, $magenta) = map (chr(27)."[".$_."m", 0, 1, 31, 32, 35);
my $ok = $green."    ok".$reset;
my $notok = $red."not ok".$reset;

if ($expected eq '-fail') {
    if ($status) {
	print "$indentTest $ok: `@args` failed on cue\n";
    } else {
	print $bold, $magenta, "\n", $test, $reset, "\n";
	print "Standard output:\n", `cat $fnameOut`;
	print "Standard error:\n", `cat $fnameErr`, $reset;
	die "$reset$indentTest $notok: `@args` succeeded (was expected to fail)\n";
    }
} else {
    if ($status) {
	print $bold, $magenta, "\n", $test, $reset, "\n";
	print "Standard output:\n", `cat $fnameOut`;
	print "Standard error:\n", `cat $fnameErr`, $reset;
	die "$reset$indentTest $notok: `@args` failed with exit code $status: $!\n";
    }

    if (-e $expected) {
	my $diff = `diff $fnameOut $expected`;

	if (length $diff) {
	    print $bold, $magenta, "\n", $test, $reset, "\n";
	    print "`@args` does not match '$expected':\n";
	    print `diff -y $fnameOut $expected`;
	    die "$indentTest $notok: `@args`\n";
	} else {
	    print "$indentTest $ok: `@args` ", ($idem ? "is idempotent" : "matches '$expected'"), "\n";
	}

    } else {
	my $actual = `cat $fnameOut`;
	chomp $actual;
	
	if ($actual eq $expected) {
	    print "$indentTest $ok: `@args` = '$expected'\n";
	} else {
	    print $bold, $magenta, "\n", $test, $reset, "\n";
	    print "`@args` does not match '$expected':\n";
	    print $actual, "\n";
	    print "Possibly '$expected' is a filename? (in which case, file not found)\n";
	    die "$indentTest $notok: `@args`\n";
	}
    }
}
