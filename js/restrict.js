#!/usr/bin/env node
// emacs mode -*-JavaScript-*-

// An automata that echoes all DNA sequences except those matching a pattern.

var fs = require('fs'),
    getopt = require('node-getopt')

// parse command-line options
var opt = getopt.create([
  ['m' , 'motif=SEQ'        , 'specify motif [ACGTN]+'],
  ['h' , 'help'             , 'display this help message']
])              // create Getopt instance
    .bindHelp()     // bind option 'help' to default action
    .parseSystem() // parse command line

var motif = opt.options.motif
if (!motif)
  throw new Error ("please specify a motif")
var motifSeq = motif.toUpperCase().split('')

// The state space is indexed (M[1],M[2]...M[motif.length-1])
// where M[K] is a binary variable that is true iff the previously-emitted K symbols match the first K characters of the motif.
// We only need to keep 2^(motif.length-1) states because the last bit is never allowed to be 1.
const alphabet = [ "A", "C", "G", "T" ]
const iupac = { A: "A", C: "C", G: "G", T: "T", W: "AT", S: "CG", M: "AC", K: "GT", R: "AG", Y: "CT", B: "CGT", D: "AGT", H: "ACT", V: "ACG", N: "ACGT" }
const initTuple = motifSeq.slice(1).fill(false)
function string2tuple (state) {
  return state.split('').map (function (c) { return c === '1' ? true : false })
}
function tuple2string (tuple) {
  return tuple.map (function (bit) { return bit ? '1' : '0' }).join('')
}
function nextTuple (tuple, tok) {
  var t = [true].concat(tuple).map (function (prevMatch, pos) {
    return prevMatch && iupac[motifSeq[pos]].indexOf(tok) >= 0
  })
  var valid = !t.pop()
  return { tok: tok, next: t, valid: valid }
}
var states = [], id2index = {}, tupleQueue = [initTuple]
while (tupleQueue.length) {
  var tuple = tupleQueue.shift(), id = tuple2string (tuple)
  if (typeof(id2index[id]) === 'undefined') {
    id2index[id] = states.length
    states.push (null)  // placeholder
    var trans = alphabet.map (function (tok) {
      return nextTuple (tuple, tok)
    }).filter (function (info) {
      return info.valid
    }).map (function (info) {
      var dest = tuple2string (info.next)
      if (!id2index[dest])
        tupleQueue.push (info.next)
      return { in: info.tok, out: info.tok, to: dest }
    })
    states[id2index[id]] = { id: id, trans: trans }
  }
}

var model = {
  state: states
}

console.log (JSON.stringify (model, null, 2))
