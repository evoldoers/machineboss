#!/usr/bin/env node
// emacs mode -*-JavaScript-*-

// Convert a PAF file (including CIGAR strings for the alignment), together with FASTA files for query & reference, to a BossMachine JSON-format training dataset
// PAF: https://github.com/lh3/miniasm/blob/master/PAF.md
// FASTA: https://en.wikipedia.org/wiki/FASTA_format

var fs = require('fs'),
    getopt = require('node-getopt'),
    readline = require('readline'),
    fasta = require('bionode-fasta')

const cigarPrefix = 'cg:'  // the prefix for the CIGAR string field at the end of each PAF line

// parse command-line options
var opt = getopt.create([
  ['p' , 'paf=PATH'         , 'PAF file'],
  ['q' , 'query=PATH'       , 'query FASTA'],
  ['t' , 'target=PATH'      , 'target FASTA'],
  ['u' , 'unaligned'        , 'do not preserve alignment'],
  ['i' , 'query-input'      , 'use query as input, target as output (default is other way around)'],
  ['s' , 'stride=N'         , 'keep only 1 in N of query sequences'],
  ['g' , 'group=G'          , 'which 1 in N to keep (0,1,2...)'],
  ['j' , 'json'             , 'JSON, not Stockholm, output'],
  ['h' , 'help'             , 'display this help message']
])              // create Getopt instance
    .bindHelp()     // bind option 'help' to default action
    .parseSystem() // parse command line

var pafFile = opt.options.paf, queryFile = opt.options.query, targetFile = opt.options.target
if (!(pafFile && queryFile && targetFile))
  throw new Error ("please specify --paf, --query, and --target")

var stride = opt.options.stride ? parseInt(opt.options.stride) : 1
var group = opt.options.group ? parseInt(opt.options.group) : 0
var json = opt.options.json
var unaligned = opt.options.unaligned
var queryInput = opt.options['query-input']

function fastaPromise (filename, stride, group) {
  var promise = new Promise (function (resolve, reject) {
    var result = {}, nSeqs = 0, nKept = 0
    console.warn ('processing ' + filename + (stride > 1 ? (' (keeping 1 in ' + stride + ' seqs)') : ''))
    fasta ({objectMode: true}, filename)
      .on ('data', function (data) {
        var id = data.id.replace(/ .*/,'')
        if (!stride || (nSeqs % stride) === group) {
          result[id] = data.seq
          ++nKept
        } else
          result[id] = null
        ++nSeqs
      })
      .on ('end', function() {
        console.warn ('finished ' + filename + ' (' + nKept + '/' + nSeqs + ' seqs)')
        resolve (result)
      })
  })
  return promise
}

function getSeq (db, name, start, end) {
  var s = parseInt (start), e = parseInt (end), seq = db[name]
  if (seq === null)
    return null
  return seq ? seq.substr (s, e - s) : false
}

var complement = {}, bases = 'acgt'.split('')
bases.forEach (function (base, n) {
  var comp = bases[bases.length-1-n]
  complement[base] = comp
  complement[base.toUpperCase()] = comp.toUpperCase()
})

function revcomp (dna) {
  return dna.split('').reverse().map (function (base) { return complement[base] }).join('')
}

function decodeCigar (query, target, cigar) {
  var align = ['','']
  var regex = /([0-9]+)([MID])/gi, rmatch, qpos = 0, tpos = 0
  while (rmatch = regex.exec(cigar)) {
    var len = parseInt (rmatch[1])
    switch (rmatch[2].toUpperCase()) {
    case 'M':
      align[0] += query.substr (qpos, len)
      align[1] += target.substr (tpos, len)
      qpos += len
      tpos += len
      break
    case 'I':
      align[0] += query.substr (qpos, len)
      align[1] += '-'.repeat(len)
      qpos += len
      break
    case 'D':
      align[0] += '-'.repeat(len)
      align[1] += target.substr (tpos, len)
      tpos += len
      break
    default:
      console.warn ("unknown expression in CIGAR string ", rmatch[1] + rmatch[2])
      break
    }
  }
  return align
}

fastaPromise (queryFile, stride, group)
  .then (function (query) {
    fastaPromise (targetFile)
      .then (function (target) {
        var nAlign = 0
        var rl = readline.createInterface ({ input: fs.createReadStream (pafFile) })
        rl.on ('line', function (line) {
          var fields = line.split ('\t')
          var q = getSeq (query, fields[0], fields[2], fields[3])
          var t = getSeq (target, fields[5], fields[7], fields[8])
          if (q && t) {
            if (fields[4] === '-')
              t = revcomp (t)
            var align
            if (!unaligned)
              fields.slice(12).forEach (function (field) {
                if (field.substr(0,cigarPrefix.length) === cigarPrefix)
                  align = decodeCigar (q, t, field.substr(cigarPrefix.length))
              })
            ++nAlign
            var qName = fields[0], tName = fields[5]
            if (json) {
              var seqPair = { input: { name: tName },
                              output: { name: qName } }
              if (align) {
                var qRow = align[0].split(''), tRow = align[1].split('')
                seqPair.alignment = qRow.map (function (qCol, n) {
                  var tCol = tRow[n]
                  return [tCol === '-' ? '' : tCol,
                          qCol === '-' ? '' : qCol]
                })
              } else {
                seqPair.input.sequence = t.split('')
                seqPair.output.sequence = q.split('')
              }
              if (queryInput) {
                var newSeqPair = { input: seqPair.output,
                                   output: seqPair.input }
                if (align)
                  newSeqPair.alignment = seqPair.alignment.map (function (col) { return [col[1], col[0]] })
                seqPair = newSeqPair
              }
              console.log ((nAlign > 1 ? ',' : '[')
                           + JSON.stringify (seqPair))
            } else {
              if (align) {
                var stock = [tName + ' ' + align[1],
                             qName + ' ' + align[0]]
                if (queryInput)
                  stock = [stock[1], stock[0]]
                console.log ('# STOCKHOLM 1.0')
                console.log (stock[0])
                console.log (stock[1])
                console.log ('//')
              }
            }
          } else {
            if (!q && q !== null)
              console.warn ('query sequence not found: ' + fields[0])
            if (!t)
              console.warn ('target sequence not found: ' + fields[5])
          }
        })
        rl.on ('close', function() {
          if (json)
            console.log (']')
        })
      })
  })
