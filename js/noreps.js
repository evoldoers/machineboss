#!/usr/bin/env node
// emacs mode -*-JavaScript-*-

var alph = ['A','C','G','T']
var state = [
  { id: 'S',
    trans: alph.map (function (base) {
      return { in: base, out: base, to: base }
    }).concat ([ { to: 'end' } ]) }
].concat (alph.map (function (base) {
  return { id: base,
           trans: alph
           .filter (function (dest) { return dest !== base })
           .map (function (base) { return { in: base, out: base, to: base } })
           .concat ([ { to: 'end' } ]) }
})).concat ([{ id: 'end', trans: [] }])

var model = { state: state }

console.log (JSON.stringify (model, null, 2))
