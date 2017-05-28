#!/usr/bin/env node

var getopt = require('node-getopt')
var negbin = require('./negbin')
var fast5 = require('./fast5')

var defaultStrand = 0
var defaultKmerLen = 6
var defaultComponents = 1

var minFracInc = 1e-5  // minimum fractional increment for EM negative binomial fit
var minExtendProb = 1e-10

var parser = getopt.create([
  ['f' , 'fast5=PATH'      , 'FAST5 input file'],
  ['s' , 'strand=N'        , 'strand (0, 1, 2; default ' + defaultStrand + ')'],
  ['g' , 'group=GROUP'     , 'group name (000, RN_001, ...)'],
  ['k' , 'kmerlen=N'       , 'kmer length (default ' + defaultKmerLen + ')'],
  ['c' , 'components=N'    , 'negative binomial mixture components (default ' + defaultComponents + ')'],
  ['h' , 'help'            , 'display this help message']
])              // create Getopt instance
.bindHelp()     // bind option 'help' to default action
var opt = parser.parseSystem(); // parse command line

function inputError(err) {
  parser.showHelp()
  console.warn (err, "\n")
  process.exit(1)
}

var filename = opt.argv[0] || opt.options['fast5'] || inputError("Please specify an input file")
var strand = opt.options.strand || defaultStrand
var group = opt.options.group || ''
var kmerLen = opt.options.kmerlen ? parseInt(opt.options.kmerlen) : defaultKmerLen
var components = opt.options.components ? parseInt(opt.options.components) : defaultComponents

var file = new fast5.File (filename)
var events = file.table_to_object (file.get_basecall_events(strand,group))

var byKmer = {}, seq = ''
var sampling_rate = file.channel_id_params.sampling_rate
var prevKmer, prevLen = 0
for (var col = 0; col < events.start.length; ++col) {
  var state = events.model_state[col], length = events.length[col], mean = events.mean[col], stdev = events.stdv[col], move = events.move[col]
  if (move)
    seq += state.substr(state.length-move).toLowerCase()
  if (seq.length >= kmerLen) {
    var kmer = seq.substr (seq.length - kmerLen)
    var m0 = Math.round (length * sampling_rate)
    var m1 = mean * m0
    var m2 = (stdev*stdev + mean*mean) * m0
    if (!byKmer[kmer])
      byKmer[kmer] = newInfo()
    var info = byKmer[kmer]
    info.m0 += m0
    info.m1 += m1
    info.m2 += m2
    if (move) {
      if (prevKmer && prevLen > 1)
        byKmer[prevKmer].lenDist[prevLen-1] = (byKmer[prevKmer].lenDist[prevLen-1] || 0) + 1
      prevLen = 0
    }
    prevKmer = kmer
    prevLen += m0
  }
}

function newInfo() { return { m0: 0, m1: 0, m2: 0, lenDist: [] } }

var alph = 'acgt'
var nKmers = Math.pow(alph.length,kmerLen)
var allKmers = new Array(nKmers).fill(0).map (function (_, n) {
  var kmer = ''
  for (var i = kmerLen - 1; i >= 0; --i)
    kmer += alph.charAt (Math.floor(n / Math.pow(alph.length,i)) % alph.length)
  return kmer
})
var gotKmers = Object.keys(byKmer).sort()
allKmers.forEach (function (kmer) {
  if (!byKmer[kmer]) {
    var equivs = []
    for (var n = 1; n < kmerLen && equivs.length === 0; ++n)
      equivs = gotKmers.filter (function (equivKmer) {
	return equivKmer.substr(n) === kmer.substr(n)
      })
    console.warn ("Missing kmer " + kmer + "; using (" + equivs.join(" ") + ")")
    var info = newInfo()
    equivs.forEach (function (equivKmer) {
      var equivInfo = byKmer[equivKmer]
      info.m0 += equivInfo.m0
      info.m1 += equivInfo.m1
      info.m2 += equivInfo.m2
      equivInfo.lenDist.forEach (function (count, len) {
        info.lenDist[len] = (info.lenDist[len] || 0) + count
      })
    })
    byKmer[kmer] = info
  }
})

var padEmitMu = 200, padEmitSigma = 50

var json = { alphabet: alph, kmerlen: kmerLen, components: components, params: { gauss: { padEmit: { mu: padEmitMu, sigma: padEmitSigma } }, rate: {}, prob: { padExtend: .5, padEnd: .5 } } }
Object.keys(byKmer).sort().forEach (function (kmer) {
  var info = byKmer[kmer]
//  console.warn("Fitting kmer " + kmer + ", length distribution [" + info.lenDist.join(",") + "]")
  var mean = info.m1 / info.m0
  var stdev = Math.sqrt (info.m2 / info.m0 - mean*mean)
  var nb = negbin.fitNegBin (info.lenDist, components, undefined, negbin.minFracInc(minFracInc))
  var rate = -Math.log (Math.max (minExtendProb, nb.pExtend))
  if (!(rate > 0 && rate < Infinity))
    throw new Error ("While fitting kmer " + kmer + ", length distribution [" + info.lenDist.join(",") + "]: rate = " + rate)
  var cptWeight = nb.rDist
  json.params.gauss["emit("+kmer+")"] = { mu: mean, sigma: stdev }
  json.params.rate["R("+kmer+")"] = rate
  if (components > 1)
    for (var cpt = 1; cpt <= components; ++cpt)
      json.params.prob["P(r="+cpt+"|"+kmer+")"] = cptWeight[cpt]
})

console.log (JSON.stringify (json))
