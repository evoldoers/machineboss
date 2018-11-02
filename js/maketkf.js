#!/usr/bin/env node

var fs = require('fs'),
    Getopt = require('node-getopt')

var getopt = Getopt.create([
  ['a' , 'alphabet=CHARS', 'specify alphabet characters explicitly'],
  ['d' , 'dna', 'DNA alphabet'],
  ['r' , 'rna', 'RNA alphabet'],
  ['p' , 'protein', 'Protein alphabet'],
  ['h' , 'help', 'display this help message']
])              // create Getopt instance
.bindHelp()     // bind option 'help' to default action
var opt = getopt.parseSystem() // parse command line

var alphabets = { dna: 'ACGT', rna: 'ACGU', protein: 'ACDEFGHIKLMNPQRSTVWY' }
Object.keys(alphabets).forEach (function (alphName) { if (opt.options[alphName]) opt.options.alphabet = alphabets[alphName] })

opt.options.alphabet || console.warn ('No alphabet specified, using DNA')
const alph = (opt.options.alphabet || alphabets.dna).split("")

function transitions (inLabel, outLabel, dest) {
    return alph.map (function (c) {
	var trans = { dest }
	if (inLabel) trans.in = c
	if (outLabel) trans.out = c
	return trans
    })
}

function not (param) {
    return {"not":param}
}

function times (expr1,expr2) {
    return {"*":[expr1,expr2]}
}

function div (expr1,expr2) {
    return {"/":[expr1,expr2]}
}

function minus (expr1,expr2) {
    return {"-":[expr1,expr2]}
}

function exp (expr) {
  return {"exp":expr}
}

function sub (c, d) {
  return "sub" + c + d
}

function eqm (c) {
  return "eqm" + c
}

var lambda = "lambda", mu = "mu", t = "t"

var pSurvived = "pSurvived", pChildless = "pChildless", pIns = "pIns", pExtinct = "pExtinct", beta = "beta", pLonger = "pLonger"
var startState = "START", deleteState = "DELETE", waitState = "WAIT", insertState = "INSERT", endState = "END"

var machine = { state:
                [{id: startState,
		  trans: [{to: insertState, weight: pIns },
			  {to: waitState, weight: not(pIns) }]},
		 {id: deleteState,
		  trans: [{to: insertState, weight: not(pExtinct) },
			  {to: waitState, weight: pExtinct }]},
                 {id: waitState,
		  trans: alph.reduce ((t,c) => t.concat (alph.map ((d) => {
                    return { in: c, out: d, to: startState, weight: times (pSurvived, sub(c,d)) }
                  })), []).concat (alph.map ((c) => {
                    return { in: c, to: deleteState, weight: not(pSurvived) }
                  })).concat ([{to: endState }])},
                 {id: insertState,
		  trans: alph.map ((c) => {return { out: c, to: startState, weight: eqm(c) } })},
		 {id: endState}],
                defs: {
                  pSurvived: exp(times(-1,times(mu,t))),
                  pChildless: exp(times(-1,times(lambda,t))),
                  beta: div(pSurvived,pChildless),
                  pLonger: div(lambda,mu),
                  pIns: div(times(pLonger,not(beta)),not(times(pLonger,beta))),
                  pExtinct: div(pIns,times(pLonger,not(pSurvived)))
                },
                cons: { norm: [alph.map (eqm)]
                        .concat (alph.map ((c) => alph.map (sub.bind (null, c)))),
                        rate: [lambda, mu, t] } }

console.log (JSON.stringify (machine, null, 2))
