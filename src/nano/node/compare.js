#!/usr/bin/env node

var getopt = require('node-getopt')
var fasta = require('bionode-fasta')
var colors = require('colors')
var fast5 = require('./fast5')

var defaultStrand = 0
var matchScore = +5, mismatchScore = -4, gapScore = -7

var opt = getopt.create([
  ['q' , 'query=PATH'      , 'FASTA query file'],
  ['r' , 'reference=PATH'  , 'FAST5 reference file'],
  ['s' , 'strand=N'        , 'strand (0, 1, 2; default ' + defaultStrand + ')'],
  ['g' , 'group=GROUP'     , 'group name (000, RN_001, ...)'],
  ['h' , 'help'            , 'display this help message']
])              // create Getopt instance
.bindHelp()     // bind option 'help' to default action
.parseSystem(); // parse command line

function inputError(err) {
    throw new Error (err)
}

var queryFilename = opt.options.query || inputError("Please specify a query file")
var refFilename = opt.options.reference || inputError("Please specify a reference file")
var strand = opt.options.strand || defaultStrand
var group = opt.options.group || ''

var fast5file = new fast5.File (refFilename)
var refSeq = fast5file.get_basecall_seq(strand,group).toLowerCase()

fasta.obj(queryFilename).on ('data', function (data) {
  var querySeq = data.seq.toLowerCase()
  var alignInfo = align (refSeq, querySeq, matchScore, mismatchScore, gapScore)
  prettyPrint (alignInfo.alignment, ['ref', 'query'])
  var matches = alignInfo.alignment.filter (function (col) { return col[0] === col[1] && col[0] !== '-' }).length
  var deletions = alignInfo.alignment.filter (function (col) { return col[1] === '-' }).length
  var insertions = alignInfo.alignment.filter (function (col) { return col[0] === '-' }).length
  var mismatches = alignInfo.alignment.length - matches - insertions - deletions
  console.log ("Score: " + alignInfo.score + " (match " + matchScore + ", mismatch "+ mismatchScore + ", gap " + gapScore + ")")
  console.log (matches + " matches")
  console.log (mismatches + " mismatches")
  console.log (insertions + " insertions")
  console.log (deletions + " deletions")
  console.log ("Reference length " + refSeq.length)
  console.log ("Query length " + querySeq.length)
})

function align(xSeq,ySeq,matchScore,mismatchScore,gapScore) {
  var xLen = xSeq.length, yLen = ySeq.length
  var matrix = new Int32Array ((xLen + 1) * (yLen + 1))
  function cellIndex(i,j) { return i*(yLen+1) + j }
  function incoming(i,j) {
    return [matrix[cellIndex(i-1,j-1)] + (xSeq.charAt(i-1) === ySeq.charAt(j-1) ? matchScore : mismatchScore),
	    matrix[cellIndex(i,j-1)] + gapScore,
	    matrix[cellIndex(i-1,j)] + gapScore]
  }
  matrix[0] = 0
  for (var i = 1; i <= xLen; ++i)
    matrix[cellIndex(i,0)] = i*gapScore
  for (var j = 1; j <= yLen; ++j)
    matrix[cellIndex(0,j)] = j*gapScore
  for (var i = 1; i <= xLen; ++i)
    for (var j = 1; j <= yLen; ++j)
      matrix[cellIndex(i,j)] = Math.max.apply (Math, incoming(i,j))
  var iTrace = xLen, jTrace = yLen, alignment = []
  while (iTrace > 0 && jTrace > 0) {
    var sc = incoming (iTrace, jTrace)
    var dir = sc.indexOf (matrix[cellIndex (iTrace, jTrace)])
    switch (dir) {
    case 0: alignment.push ([xSeq.charAt(--iTrace),ySeq.charAt(--jTrace)]); break;
    case 1: alignment.push (['-',ySeq.charAt(--jTrace)]); break;
    case 2: alignment.push ([xSeq.charAt(--iTrace),'-']); break;
    default: throw new Error("traceback error"); break;
    }
  }
  while (iTrace) alignment.push ([xSeq.charAt(--iTrace),'-'])
  while (jTrace) alignment.push (['-',ySeq.charAt(--jTrace)])
  alignment.reverse()
  return { alignment: alignment,
	   score: matrix[cellIndex(xLen,yLen)] }
}

function prettyPrint (alignment, labels, refRowIndex) {
  refRowIndex = refRowIndex || 0
  alignment.forEach (function (colData, colIndex) {
    if (colData.length != labels.length)
      throw new Error ("Column #" + (colIndex+1) + " has " + colData.length + " rows, but there are " + labels.length + " row labels")
  })
  var labelCols = Math.max.apply (Math, labels.map (function (label) { return label.length }))
  var alignCols = process.stdout.columns - labelCols - 1
  for (var colStart = 0; colStart < alignment.length; colStart += alignCols) {
    labels.forEach (function (rowLabel, rowIndex) {
      process.stdout.write (rowLabel + new Array(labelCols+1-rowLabel.length).fill(' ').join(''))
      for (var colIndex = colStart; colIndex < colStart + alignCols && colIndex < alignment.length; ++colIndex) {
	var alignChar = alignment[colIndex][rowIndex], refChar = alignment[colIndex][refRowIndex]
	process.stdout.write (alignChar === refChar ? alignChar : alignChar.inverse)
      }
      process.stdout.write ("\n")
    })
    process.stdout.write ("\n")
  }
}
