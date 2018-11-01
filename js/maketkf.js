#!/usr/bin/env node

var fs = require('fs'),
    Getopt = require('node-getopt')

var getopt = Getopt.create([
    ['a' , 'alphabet=STRING'  , 'alphabet'],
    ['h' , 'help'             , 'display this help message']
])              // create Getopt instance
.bindHelp()     // bind option 'help' to default action
var opt = getopt.parseSystem() // parse command line

opt.options.alphabet || console.warn ('No alphabet specified, using DNA')
const alph = (opt.options.alphabet || 'ACGT').split("")

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
function insOpen() {
}

var alpha = "alpha", beta = "beta", gamma = "gamma", rho = "rho", kappa = "kappa"
var machine = { state:
                [{id: "S",
		  trans: [{to: "I", weight: beta },
			  {to: "W", weight: not(beta) }]},
		 {id: "D",
		  trans: [{to: "I", weight: gamma },
			  {to: "W", weight: not(gamma) }]},
                 {id: "W",
		  trans: alph.reduce ((t,c) => t.concat (alph.map ((d) => {
                    return { in: c, out: d, to: "S", weight: times (alpha, sub(c,d)) }
                  })), []).concat (alph.map ((c) => {
                    return { in: c, to: "D", weight: not(alpha) }
                  })).concat ([{to: "E" }])},
                 {id: "I",
		  trans: alph.map ((c) => {return { out: c, to: "S", weight: eqm(c) } })},
		 {id: "E"}],
                defs: {
                  kappa: div(lambda,mu),
                  rho: exp(times(t,minus(lambda,mu))),
                  alpha: exp(times(-1,times(mu,t))),
                  beta: times(kappa,div(not(rho),not(times(kappa,rho)))),
                  gamma: not(div(beta,times(not(alpha),kappa)))
                },
                cons: { norm: [alph.map ((c) => "eqm"+c)]
                        .concat (alph.map ((c) => alph.map ((d)=>"sub"+c+d))),
                        rate: [lambda, mu, t] } }

console.log (JSON.stringify (machine, null, 2))
