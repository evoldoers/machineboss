#!/usr/bin/env node

var Getopt = require('node-getopt')

var getopt = Getopt.create([
    ['u' , 'upper', 'convert to upper instead of lower case'],
    ['h' , 'help', 'display this help message']
])              // create Getopt instance
.bindHelp()     // bind option 'help' to default action
var opt = getopt.parseSystem() // parse command line

var space = 32, tilde = 126, a_upper = 65, a_lower = 97, alph_size = 26;
var opts = (opt.options.upper
	    ? { begin: a_lower, to: a_upper }
	    : { begin: a_upper, to: a_lower })
opts.end = opts.begin + alph_size - 1
opts.delta = opts.to - opts.begin
var json = {
  state: [{
    n: 0,
    trans: new Array (tilde + 1 - space).fill(0).map ((_c, n) => {
      const cc = n + space;
      const in_c = String.fromCharCode (cc);
      const out_c = String.fromCharCode ((cc >= opts.begin && cc <= opts.end) ? (cc + opts.delta) : cc);
      return { to: 0, in: in_c, out: out_c }
    })
  }]
};

console.log (JSON.stringify (json, null, 2))
