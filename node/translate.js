#!/usr/bin/env node

var fs = require('fs')
var path = require('path')
var Getopt = require('node-getopt')

var defaultCodonPath = path.resolve(__dirname, '../data/codon-usage.txt')
var defaultName = "translate"

var getopt = Getopt.create([
    ['c' , 'codon=PATH'    , 'codon translation & relative frequency table (default is "' + defaultCodonPath + '")'],
    ['n' , 'name=STRING'   , 'name (default is "' + defaultName + '")'],
    ['p' , 'pretty'],
    ['h' , 'help'             , 'display this help message']
])              // create Getopt instance
.bindHelp()     // bind option 'help' to default action
var opt = getopt.parseSystem() // parse command line

function inputError(err) {
    getopt.showHelp()
    console.warn (err)
    process.exit(1)
}

var codonPath = opt.options.codon || defaultCodonPath
var name = opt.options.name || defaultName

var aa2codons = {}, codon2aa = {}, codonFreq = {}, codons = [], cod23 = {}, cod3 = {}
fs.readFileSync(codonPath).toString().split("\n").forEach ((line) => {
    var [codon, aa, freq] = line.split(" ")
    if (codon.length == 3 && aa.length == 1 && aa !== '*') {
	codon = codon.toUpperCase()
	aa = aa.toUpperCase()
	aa2codons[aa] = aa2codons[aa] || []
	aa2codons[aa].push (codon)
	codonFreq[codon] = parseFloat (freq)
	codon2aa[codon] = aa
	codons.push (codon)
	cod23[codon.substr(1)] = true
	cod3[codon.substr(2)] = true
    }
})
var aa = Object.keys(aa2codons).sort()

function makeParam (aa, codon) {
    return aa + '_' + codon
}

var machine =
    { state:
      [{id:name+'-S',
	trans: codons.map ((cod) => { return { in: codon2aa[cod], out: cod.charAt(0), to: cod.substr(1), weight: aa2codons[codon2aa[cod]].length == 1 ? undefined : makeParam (codon2aa[cod], cod) } })
	.concat ([{in:'base',out:'base',to:name+'-S'},
		  {in:'intron',out:'intron',to:name+'-S'},
		  {to:name+'-E'}]) }]
      .concat (Object.keys(cod23).sort().map ((c23) => { return { id: c23, trans: [ { out: c23.charAt(0), to: c23.substr(1) } ] } }))
      .concat (Object.keys(cod3).sort().map ((c3) => { return { id: c3, trans: [ { out: c3, to: name+'-S' } ] } }))
      .concat ([{id:name+'-E'}]) }

var cons = { norm: Object.keys(aa2codons).sort().filter((aa) => (aa2codons[aa].length > 1)).map ((aa) => { return aa2codons[aa].map ((c) => { return makeParam (aa, c) }) }) }
var param = {};
Object.keys(aa2codons).sort().map ((aa) => {
    if (aa2codons[aa].length > 1)
	aa2codons[aa].forEach ((c) => {
	    param[makeParam(aa,c)] = codonFreq[c]
	})
})

fs.writeFileSync ('preset/'+name+'.json', JSON.stringify (machine))
fs.writeFileSync ('constraints/'+name+'.json', JSON.stringify (cons))
fs.writeFileSync ('params/'+name+'.json', JSON.stringify (param))
