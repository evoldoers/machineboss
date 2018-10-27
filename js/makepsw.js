#!/usr/bin/env node

var fs = require('fs'),
    Getopt = require('node-getopt')

var getopt = Getopt.create([
    ['a' , 'alphabet=STRING'  , 'alphabet'],
    ['n' , 'name=STRING'      , 'name'],
    ['m' , 'mix=N'            , 'no. of components to mixture geometric indel-length distribution'],
    ['w' , 'write'            , 'write preset & constraints files'],
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
var mixCpts = opt.options.mix

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

function iota (n) {
  // return [1,1...n] if n is defined, or [''] if not
  return (typeof(n) === 'undefined'
          ? ['']
          : new Array(parseInt(n)).fill(0).map (function(_val,k) { return k + 1 }))
}

var iotaMix = iota (mixCpts)
var machine = { state: [{id: name+"-S",
			 trans: iotaMix.map ((k) => {return {to: name+"-I"+k, weight: "gapOpen"+k}} )
			 .concat ([{ to: name+"-W", weight: (mixCpts ? "notGapOpen" : not("gapOpen")) }])}]
		.concat (iotaMix.map ((k) => { return {id: name+"-J"+k,
			                               trans: [{to: name+"-I"+k, weight: "gapExtend"},
				                               {to: name+"-W", weight: not("gapExtend")}]} }))
		.concat ([{id: name+"-W",
		           trans: [{ to: name+"-M", weight: (mixCpts ? "notGapOpen" : not("gapOpen")) }]
                           .concat (iotaMix.map ((k) => { return {to: name+"-D"+k, weight: "gapOpen"} } ))
                          }])
                .concat (iotaMix.map ((k) => { return {id: name+"-X"+k,
			                               trans: [{to: name+"-D"+k, weight: "gapExtend"},
				                               {to: name+"-M", weight: not("gapExtend")}]} } ))
                .concat (iotaMix.map ((k) => { return {id: name+"-I"+k,
			                               trans: alph.map ((c) => {return { out: c, to: name+"-J"+k, weight: "eqm"+c } })} }))
		.concat ([{id: name+"-M",
			   trans: [{to: name+"-E"}]
			   .concat (alph.reduce ((t,c) => t.concat (alph.map ((d) => {return { in: c, out: d, to: name+"-S", weight: "sub"+c+d }})), []))}])
		.concat (iotaMix.map ((k) => { return {id: name+"-D"+k,
			                               trans: [{to: name+"-E"}]
			                               .concat (alph.map ((c) => {return { in: c, to: name+"-X"+k }}))} } ))
		.concat ([{id: name+"-E"} ]),
                cons: { prob: (mixCpts
                               ? iotaMix.map ((k) => "gapExtend"+k)
                               : ["gapOpen","gapExtend"]),
                        norm: [alph.map ((c) => "eqm"+c)]
                        .concat (alph.map ((c) => alph.map ((d)=>"sub"+c+d)))
                        .concat (mixCpts
                                 ? [iotaMix.map((k)=>"gapOpen"+k).concat("notGapOpen")]
                                 : []) } }

if (opt.options.write) {
  fs.writeFileSync ("preset/"+name+".json", JSON.stringify (machine, null, opt.options.pretty ? 2 : null))
  fs.writeFileSync ("constraints/"+name+".json", JSON.stringify (machine.cons, null, opt.options.pretty ? 2 : null))
} else {
  console.log (JSON.stringify (machine, null, 2))
}
