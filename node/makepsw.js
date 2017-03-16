#!/usr/bin/env node

var fs = require('fs'),
    Getopt = require('node-getopt')

var getopt = Getopt.create([
    ['a' , 'alphabet=STRING'  , 'alphabet'],
    ['c' , 'constraints'      , 'make constraints file'],
    ['p' , 'pretty'],
    ['h' , 'help'             , 'display this help message']
])              // create Getopt instance
.bindHelp()     // bind option 'help' to default action
var opt = getopt.parseSystem() // parse command line

function inputError(err) {
    getopt.showHelp()
    console.warn (err)
    process.exit(1)
}

opt.options.alphabet || inputError ("Please specify an alphabet")
var alph = opt.options.alphabet.split("")

function transitions (inLabel, outLabel, dest) {
    return alph.map (function (c) {
	var trans = { dest }
	if (inLabel) trans.in = c
	if (outLabel) trans.out = c
	return trans
    })
}

function not (param) {
    return {"-":[true,param]}
}

function times (expr1,expr2) {
    return {"*":[expr1,expr2]}
}

var machine = { state: [{id: "S",
			 trans: [{to: "I", weight: "gapOpen"},
				 {to: "W", weight: not("gapOpen")}]},
			{id: "I",
			 trans: alph.map ((c) => {return { out: c, to: "J", weight: "eqm"+c } })},
			{id: "J",
			 trans: [{to: "I", weight: "gapExtend"},
				 {to: "W", weight: not("gapExtend")}]},
			{id: "W",
			 trans: [{to: "M", weight: "gapOpen"},
				 {to: "D", weight: not("gapOpen")}]},
			{id: "M",
			 trans: [{to: "E"}]
			 .concat (alph.reduce ((t,c) => t.concat (alph.map ((d) => {return { in: c, out: d, to: "S", weight: "sub"+c+d }})), []))},
			{id: "D",
			 trans: [{to: "E"}]
			 .concat (alph.map ((c) => {return { in: c, to: "X" }}))},
			{id: "X",
			 trans: [{to: "D", weight: "gapExtend"},
				 {to: "M", weight: not("gapExtend")}]},
		       {id: "E"} ] }

var constraints = [{prob:["gapOpen","gapExtend"],
		    norm:alph.map((c) => alph.map((d)=>"sub"+c+d))}]

console.log (JSON.stringify (opt.options.constraints ? constraints : machine, null, opt.options.pretty ? 2 : null))
