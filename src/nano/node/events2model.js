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
var events = file.table_to_object (file.get_basecall_events(strand,group))

var byState = {}
var sampling_rate = file.channel_id_params.sampling_rate
for (var col = 0; col < events.start.length; ++col) {
  var state = events.model_state[col], length = events.length[col], mean = events.mean[col], stdev = events.stdv[col], move = events.move[col]
  var m0 = length * sampling_rate
  var m1 = mean * m0
  var m2 = (stdev*stdev + mean*mean) * m0
  if (!byState[state])
    byState[state] = { m0: 0, m1: 0, m2: 0, moves: 0 }
  var info = byState[state]
  info.m0 += m0
  info.m1 += m1
  info.m2 += m2
  if (move)
    ++info.moves
}

var padEmitMu = 200, padEmitSigma = 50

var json = { alphabet: "acgt", kmerlen: null, components: 1, params: { gauss: { padEmit: { mu: padEmitMu, sigma: padEmitSigma } }, rate: {}, prob: { padExtend: .5, padEnd: .5 } } }
Object.keys(byState).forEach (function (kmer) {
  var info = byState[kmer]
  var mean = info.m1 / info.m0
  var stdev = Math.sqrt (info.m2 / info.m0 - mean*mean)
  json.kmerlen = kmer.length
  json.params.gauss["emit("+kmer+")"] = { mu: mean, sigma: stdev }
  json.params.rate["R(move|"+kmer+",cpt1)"] = info.moves / info.m0
})

console.log (JSON.stringify (json))
