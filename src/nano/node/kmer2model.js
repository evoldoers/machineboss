#!/usr/bin/env node

var fs = require('fs')

var filename = process.argv[2]
var lines = fs.readFileSync(filename).toString().split("\n").slice(1)

var json = { alphabet: "acgt", components: 1, kmerlen: 6, params: { gauss: { padEmit: { mu: 0, sigma: 100 } }, rate: {}, prob: { padExtend: .5, padEnd: .5 } } }
lines.forEach (function (line) {
  var fields = line.split("\t")
  var kmer = fields[0].toLowerCase(), level_mean = parseFloat(fields[1]), sd_mean = parseFloat(fields[3])
  json.params.gauss["emit("+kmer+")"] = { mu: level_mean, sigma: sd_mean }
  json.params.rate["R(move|"+kmer+",cpt1)"] = 1
})

console.log (JSON.stringify (json))
