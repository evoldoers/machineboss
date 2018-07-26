#!/usr/bin/env node

var fs = require('fs'),
    path = require('path'),
    Getopt = require('node-getopt')

var getopt = Getopt.create([
  ['m' , 'module=FILE'      , 'specify bossmachine-generated module'],
  ['h' , 'help'             , 'display this help message']
])              // create Getopt instance
.bindHelp()     // bind option 'help' to default action
var opt = getopt.parseSystem() // parse command line

var argv = opt.argv, argc = argv.length
if (argc != 2 && argc != 3) {
  console.warn ("Usage: " + path.basename (__filename) + " [--help] inputProfile.csv outputProfile.csv [params.json]")
  process.exit(1)
}

var module = opt.options.module
if (!module) {
  console.warn ("You must specify a module with --module")
  process.exit(1)
}
var computeForward = require('./'+module).computeForward

function readCSV (filename) {
  var rows = fs.readFileSync(filename).toString().split('\n').filter (function (line) { return line.length }).map (function (line) { return line.split(',') })
  return { header: rows[0], row: rows.slice(1).map (function (row) { return row.map (parseFloat) }) }
}

var inProf = readCSV (argv[0])
var outProf = readCSV (argv[1])
var params
if (argc > 2) {
  params = JSON.parse (fs.readFileSync (argv[2]).toString())
} else
  params = {}

console.log (JSON.stringify ([computeForward (inProf.row, outProf.row, params)]))
