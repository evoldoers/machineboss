#!/usr/bin/env node

var getopt = require('node-getopt')
var fast5 = require('./fast5')

var defaultStrand = 0

var opt = getopt.create([
  ['f' , 'fast5=PATH'      , 'FAST5 input file'],
  ['s' , 'strand=N'        , 'strand (0, 1, 2; default ' + defaultStrand + ')'],
  ['g' , 'group=GROUP'     , 'group name (000, RN_001, ...)'],
  ['h' , 'help'            , 'display this help message']
])              // create Getopt instance
.bindHelp()     // bind option 'help' to default action
.parseSystem(); // parse command line

function inputError(err) {
    throw new Error (err)
}

var filename = opt.argv[0] || opt.options['fast5'] || inputError("Please specify an input file")
var strand = opt.options.strand || defaultStrand
var group = opt.options.group || ''

var file = new fast5.File (filename)
console.log(file)
