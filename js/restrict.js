#!/usr/bin/env node
// emacs mode -*-JavaScript-*-

// An automata that echoes all DNA sequences except those matching a pattern.

var fs = require('fs'),
    getopt = require('node-getopt')

// parse command-line options
var opt = getopt.create([
  ['m' , 'motif=SEQ'        , 'specify motif [ACGTN]+'],
  ['f' , 'forward'          , 'do not check reverse strand'],
  ['r' , 'repeats'          , 'allow homopolymer repeats'],
  ['h' , 'help'             , 'display this help message']
])              // create Getopt instance
    .bindHelp()     // bind option 'help' to default action
    .parseSystem() // parse command line

var motif = opt.options.motif || 'Z', forward = opt.options.forward, repeats = opt.options.repeats
var motifSeq = motif.toUpperCase().split('')

// The state space is indexed (P,M[1],M[2]...M[motif.length-1])
// where
//  P is the previous nucleotide (or the empty string if in the initial state),
//  M[K] is a binary variable that is true iff the previously-emitted K symbols match the first K characters of the motif.
// We only need to keep 2^(motif.length-1) states because the last bit is never allowed to be 1.
const alphabet = [ "A", "C", "G", "T" ], complement = { A:"T", C:"G", G:"C", T:"A" }
const iupac = { A: "A", C: "C", G: "G", T: "T", W: "AT", S: "CG", M: "AC", K: "GT", R: "AG", Y: "CT", B: "CGT", D: "AGT", H: "ACT", V: "ACG", N: "ACGT", Z: "" }
function motifMatch (pos, queryChar) { return iupac[motifSeq[pos]].indexOf(queryChar) >= 0 }
const tupleChars = [ '_', 'f', 'r', 'b' ], tupleCharIndex = { _:0, f:1, r:2, b:3 }
const initTuple = [''].concat (motifSeq.slice(1).fill(0))
const tupleRegex = new RegExp ('([frb])([0-9]+)','g')
function string2tuple (state) {
  var tuple = initTuple.slice(0)
  var match
  while (match = tupleRegex.exec (state)) {
    var flags = tupleCharIndex[match[1]], pos = parseInt(match[2]) - 1
    tuple[pos] = flags
  }
  return tuple
}
function tuple2string (tuple) {
  return (tuple[0] || '_') + tuple.slice(1).map (function (flags, pos) {
    return flags ? (tupleChars[flags] + (pos+1)) : ''
  }).join('')
}
function nextTuple (tuple, tok) {
  var t = [forward ? 1 : 3].concat(tuple.slice(1)).map (function (prevMatch, pos) {
    var prevFwdMatch = (prevMatch & 1) ? true : false
    var prevRevMatch = (prevMatch & 2) ? true : false
    var fwdMatch = prevFwdMatch && motifMatch (pos, tok)
    var revMatch = prevRevMatch && motifMatch (motifSeq.length-1-pos, complement[tok])
    return (fwdMatch ? 1 : 0) | (revMatch ? 2 : 0)
  })
  var valid = !t.pop() && (repeats || tok !== tuple[0])
  return { tok: tok, next: [tok].concat(t), valid: valid }
}
var states = [], id2index = {}, tupleQueue = [initTuple], incoming = { start: {} }
while (tupleQueue.length) {
  var tuple = tupleQueue.shift(), id = tuple2string (tuple)
  if (typeof(id2index[id]) === 'undefined') {
    id2index[id] = states.length
    incoming[id] = incoming[id] || {}
    states.push (null)  // placeholder
    var preTrans = alphabet.map (function (tok) {
      return nextTuple (tuple, tok)
    }).filter (function (info) {
      return info.valid
    })
    var trans = preTrans.map (function (info, pos) {
      var dest = tuple2string (info.next)
      if (!id2index[dest]) {
        tupleQueue.push (info.next)
        incoming[dest] = {}
      }
      incoming[dest][id] = true
      return { in: pos + '_' + preTrans.length,
               out: info.tok,
               to: dest }
    })
    states[id2index[id]] = { id: id, trans: trans.concat ([{ to: 'end' }]) }
  }
}
id2index['end'] = states.length
incoming['end'] = {}
states.forEach (function (state) { incoming['end'][state.id] = true })
states.push ({ id: 'end', trans: [] })

var reachable = {}, sinkQueue = ['end']
while (sinkQueue.length) {
  var sink = sinkQueue.shift()
  reachable[sink] = true
  Object.keys (incoming[sink]).forEach (function (source) {
    if (!reachable[source])
      sinkQueue.push (source)
  })
}

states = states.filter (function (state) {
  return reachable[state.id]
}).map (function (state) {
  return { id: state.id,
           trans: state.trans.filter (function (t) { return reachable[t.to] }) }
})

var model = {
  state: states
}

console.log (JSON.stringify (model, null, 2))
