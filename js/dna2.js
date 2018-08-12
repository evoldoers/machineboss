var alph = "ACGT".split("")
var name = "dna2"

function not (param) {
    return {"-":[true,param]}
}

function times (expr1,expr2) {
    return {"*":[expr1,expr2]}
}

var contexts = alph.concat (['X'])

var startState = { name: 'start', trans: [] },
    endState = { name: 'end' }

function matchState (leftContext, rightContext) { return 'M' + leftContext + rightContext }
function insertState (leftContext, rightContext) { return 'I' + leftContext + rightContext }
function deleteState (leftContext, rightContext) { return 'D' + leftContext + rightContext }

function subProb (i, j, leftContext, rightContext) { return 'q' + i + j + '_' + leftContext + rightContext }
function insOpen (leftContext, rightContext) { return 'i' + '_' + leftContext + rightContext }
function insExtend (leftContext, rightContext) { return 'j' + '_' + leftContext + rightContext }
function insEmit (i, leftContext, rightContext) { return 'k' + i + '_' + leftContext + rightContext }
function delOpen (leftContext, rightContext) { return 'd' + '_' + leftContext + rightContext }
function delExtend (leftContext, rightContext) { return 'e' + '_' + leftContext + rightContext }
function delConfirm (j, leftContext, rightContext) { return 'f' + j + '_' + leftContext + rightContext }

var states = [startState]
contexts.forEach (function (leftContext) {
  contexts.forEach (function (rightContext) {
    // declare states & transitions
  })
})
states.push (endState)

var norms = []
alph.forEach (function (leftContext) {
  alph.forEach (function (rightContext) {
    // declare parameter groups
  })
})

var machine = { state: states,
                cons: { norm: norms } }

console.log (JSON.stringify (machine))
