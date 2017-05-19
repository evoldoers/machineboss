#!/usr/bin/env node

var fs = require('fs'),
    path = require('path'),
    getopt = require('node-getopt'),
    MersenneTwister = require('mersennetwister'),
    jStat = require('jStat').jStat,
    assert = require('assert'),
    fasta = require('bionode-fasta')

var defaultSeed = 123456789

var opt = getopt.create([
    ['f' , 'fasta=PATH'       , 'input sequence (FASTA)'],
    ['m' , 'model=PATH'       , 'model parameters (JSON)'],
    ['s' , 'shift=N'          , 'shift'],
    ['c' , 'scale=N'          , 'scale'],
    ['r' , 'rate=N'           , 'rate'],
    ['R' , 'rnd-seed=N'       , 'seed random number generator (default=' + defaultSeed + ')'],
    ['h' , 'help'             , 'display this help message']
])              // create Getopt instance
.bindHelp()     // bind option 'help' to default action
.parseSystem(); // parse command line

function inputError(err) {
    throw new Error (err)
}

var fastaPath = opt.options['fasta'] || inputError("You must specify a FASTA sequence file")
var modelPath = opt.options['model'] || inputError("You must specify a JSON model file")

var seed = parseInt(opt.options['rnd-seed']) || defaultSeed
var generator = new MersenneTwister (seed)
Math.random = generator.random.bind(generator)

var model = JSON.parse (fs.readFileSync (modelPath))
if (model.components !== 1)
  throw new Error ("Can only deal with 1 component at the moment")

var scale = parseFloat(opt.options['scale']) || 1
var shift = parseFloat(opt.options['shift']) || 0
var rate = parseFloat(opt.options['rate']) || 1

fasta.obj(fastaPath).on ('data', function (data) {
  emitPad()
  for (var pos = 0; pos <= data.seq.length - model.kmerlen; ++pos) {
    var kmer = data.seq.substr(pos,model.kmerlen).toLowerCase()
    var rateParam = "R(move|" + kmer + ",cpt1)"
    var emitParam = "emit(" + kmer + ")"
    var moveRate = model.params.rate[rateParam]
    var stayProb = Math.exp (-rate * moveRate)
    do {
      emit (emitParam)
    } while (generator.random() < stayProb)
  }
  emitPad()
})

function emitPad() {
  while (generator.random() < model.params.prob.padExtend) {
    emit ("padEmit")
  }
}

function emit (label) {
  var params = model.params.gauss[label]
  var x = jStat.normal.sample (params.mu, params.sigma)
  console.log (scale * (x + shift) + '  # ' + label)
}
