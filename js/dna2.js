var alph = "ACGT".split("")
var name = "dna2"

function not (param) {
    return {"-":[true,param]}
}

function times (expr1,expr2) {
    return {"*":[expr1,expr2]}
}

var startState = { name: 'start', trans: [] },
    endState = { name: 'end' }

function matchState (leftContext, rightContext) { return 'M' + leftContext + rightContext }
function insertState (leftContext, rightContext) { return 'I' + leftContext + rightContext }
function deleteState (leftContext, rightContext) { return 'D' + leftContext + rightContext }

function eqmProb (i) { return 'p' + i }
function subProb (i, j, leftContext, rightContext) { return 'sub' + i + j + '_' + leftContext + rightContext }
function insOpen (leftContext, rightContext) { return 'io' + '_' + leftContext + rightContext }
function insExtend (leftContext, rightContext) { return 'ix' + '_' + leftContext + rightContext }
function insChar (i, leftContext, rightContext) { return 'ic' + i + '_' + leftContext + rightContext }
function delOpen (leftContext, rightContext) { return 'do' + '_' + leftContext + rightContext }
function delExtend (leftContext, rightContext) { return 'dx' + '_' + leftContext + rightContext }
function delConfirm (j, leftContext, rightContext) { return 'dc' + j + '_' + leftContext + rightContext }

var states = [startState]
var norms = [], probs = []
alph.forEach (function (l) {
  alph.forEach (function (r) {
    // declare states & transitions
    startState.trans.push ({ to: matchState(l,r),
                             weight: eqmProb(l) })
    states.push ({ name: matchState(l,r),
                   trans: [] },
                 { name: insertState(l,r),
                   trans: [] },
                 { name: deleteState(l,r),
                   trans: [] })
    // declare parameter groups
  })
})
states.push (endState)

var machine = { state: states,
                cons: { norm: norms,
                        prob: probs } }

console.log (JSON.stringify (machine))
