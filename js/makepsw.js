#!/usr/bin/env node

var fs = require('fs'),
    Getopt = require('node-getopt')

var getopt = Getopt.create([
    ['a' , 'alphabet=STRING'  , 'alphabet'],
    ['n' , 'name=STRING'      , 'name'],
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
opt.options.name || inputError ("Please specify a model name")
var alph = opt.options.alphabet.split("")
var name = opt.options.name

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

var machine = { state: [{id: name+"-S",
			 trans: [{to: name+"-I", weight: "gapOpen"},
				 {to: name+"-W", weight: not("gapOpen")}]},
			{id: name+"-J",
			 trans: [{to: name+"-I", weight: "gapExtend"},
				 {to: name+"-W", weight: not("gapExtend")}]},
			{id: name+"-W",
			 trans: [{to: name+"-M", weight: not("gapOpen")},
				 {to: name+"-D", weight: "gapOpen"}]},
			{id: name+"-X",
			 trans: [{to: name+"-D", weight: "gapExtend"},
				 {to: name+"-M", weight: not("gapExtend")}]},
			{id: name+"-I",
			 trans: alph.map ((c) => {return { out: c, to: name+"-J", weight: "eqm"+c } })},
			{id: name+"-M",
			 trans: [{to: name+"-E"}]
			 .concat (alph.reduce ((t,c) => t.concat (alph.map ((d) => {return { in: c, out: d, to: name+"-S", weight: "sub"+c+d }})), []))},
			{id: name+"-D",
			 trans: [{to: name+"-E"}]
			 .concat (alph.map ((c) => {return { in: c, to: name+"-X" }}))},
		        {id: name+"-E"} ],
                cons: { prob:["gapOpen","gapExtend"],
		        norm: [alph.map ((c) => "eqm"+c)]
                        .concat (alph.map ((c) => alph.map ((d)=>"sub"+c+d)))} }

fs.writeFileSync ("preset/"+name+".json", JSON.stringify (machine, null, opt.options.pretty ? 2 : null))
fs.writeFileSync ("constraints/"+name+".json", JSON.stringify (machine.cons, null, opt.options.pretty ? 2 : null))
