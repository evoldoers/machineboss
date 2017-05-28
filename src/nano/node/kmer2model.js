#!/usr/bin/env node

var fs = require('fs')

var filename = process.argv[2]
var lines = fs.readFileSync(filename).toString().split("\n").slice(1)

var padEmitMu = 200, padEmitSigma = 50

var json = { alphabet: "acgt", kmerlen: null, components: 1, params: { gauss: { padEmit: { mu: padEmitMu, sigma: padEmitSigma } }, rate: {}, prob: { padExtend: .5, padEnd: .5 } } }
lines.forEach (function (line) {
  var fields = line.split("\t")
  var kmer = fields[0].toLowerCase(), level_mean = parseFloat(fields[1]), sd_mean = parseFloat(fields[3])
  if (kmer.length) {
    json.kmerlen = kmer.length
    json.params.gauss["emit("+kmer+")"] = { mu: level_mean, sigma: sd_mean }
    json.params.rate["R("+kmer+")"] = 1
  }
})

console.log (JSON.stringify (json))
