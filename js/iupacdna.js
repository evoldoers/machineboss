#!/usr/bin/env node

const alph = { A: 'A',
               C: 'C',
               G: 'G',
               T: 'T',
               R: 'AG',
	       Y: 'CT',
               S: 'GC',
               W: 'AT',
               K: 'GT',
               M: 'AC',
               B: 'CGT',
               D: 'AGT',
               H: 'ACT',
               V: 'ACG',
               N: 'ACGT' }

const machine = { state: [{ n: 0,
                            trans: Object.keys(alph).reduce ((trans, c) => trans.concat (alph[c].split('').map (d => { return { to: 0, in: c, out: d } })), []) }] }

console.log (JSON.stringify (machine))
