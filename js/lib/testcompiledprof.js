#!/usr/bin/env node

var fs = require('fs'),
    path = require('path'),
    Getopt = require('node-getopt'),
    sp = require('./softplus')

var getopt = Getopt.create([
  ['p' , 'params=FILE'      , 'specify parameters as JSON file'],
  ['x' , 'inprof=FILE'      , 'specify input profile as CSV file'],
  ['y' , 'outprof=FILE'     , 'specify output profile as CSV file'],
  ['i' , 'inseq=STRING'     , 'specify input sequence as string'],
  ['o' , 'outseq=STRING'    , 'specify output sequence as string'],
  ['h' , 'help'             , 'display this help message']
])              // create Getopt instance
.bindHelp()     // bind option 'help' to default action
var opt = getopt.parseSystem() // parse command line

var input, output, params

function readCSV (filename) {
  var rows = fs.readFileSync(filename).toString().split('\n').filter (function (line) { return line.length }).map (function (line) { return line.split(',') })
  return { header: rows[0], row: rows.slice(1).map (function (row) { return row.map (parseFloat) }) }
}

if (opt.options.inprof)
  input = readCSV (opt.options.inprof).row
else if (opt.options.inseq)
  input = opt.options.inseq
else
  throw new Error ("You must specify an input sequence with --inprof or --inseq")

if (opt.options.outprof)
  output = readCSV (opt.options.outprof).row
else if (opt.options.outseq)
  output = opt.options.outseq
else
  throw new Error ("You must specify an output sequence with --outprof or --outseq")

if (opt.options.params)
  params = JSON.parse (fs.readFileSync (opt.options.params).toString())
else
  params = {}

console.log (JSON.stringify ([computeForward (input, output, params)]))
