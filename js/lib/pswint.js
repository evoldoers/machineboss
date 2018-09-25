var alph = "ACDEFGHIKLMNPQRSTVWY".split("")
var name = "pswint"

function transitions (inLabel, outLabel, dest) {
    return alph.map (function (c) {
	var trans = { dest }
	if (inLabel) trans.in = c
	if (outLabel) trans.out = c
	return trans
    })
}

function not (param) {
    return {"-":[true,param]}
}

function times (expr1,expr2) {
    return {"*":[expr1,expr2]}
}

function intronStates (prefix) {
    return [{ id: name+"-"+prefix+"-intron", trans: [{ to: name+"-"+prefix+"-BB", out: "intron", weight: {"/":[1,3]} },
						     { to: name+"-"+prefix+"-IB", out: "base", weight: {"/":[1,3]} },
						     { to: name+"-"+prefix+"-BI", out: "base", weight: {"/":[1,3]} }] },
            { id: name+"-"+prefix+"-BB", trans: [{ out: "base", to: name+"-"+prefix+"-B" }] },
            { id: name+"-"+prefix+"-B", trans: [{ out: "base", to: name+"-"+prefix }] },
            { id: name+"-"+prefix+"-IB", trans: [{ out: "intron", to: name+"-"+prefix+"-B" }] },
            { id: name+"-"+prefix+"-BI", trans: [{ out: "base", to: name+"-"+prefix+"-I" }] },
            { id: name+"-"+prefix+"-I", trans: [{ out: "intron", to: name+"-"+prefix }] }]
}

function machine (pswFlag) {
  var startState = pswFlag ? (name+"-S") : (name+"-M")
  var cons = { prob: ["intron"] }
  if (pswFlag) {
    cons.prob = ["gapOpen", "gapExtend"].concat (cons.prob)
    cons.norm = [alph.map ((c) => "eqm"+c)]
      .concat (alph.map ((c) => alph.map ((d)=>"sub"+c+d)))
  }
  return { state: ((pswFlag
                    ? [{id: name+"-S",
		        trans: [{to: name+"-I", weight: "gapOpen"},
			        {to: name+"-W", weight: not("gapOpen")}]},
		       {id: name+"-I",
		        trans: alph.map ((c) => { return { out: c, to: name+"-J", weight: times(not("intron"),"eqm"+c) } })
		        .concat ([{ to: name+"-I-intron", weight: "intron" }])},
		       {id: name+"-J",
		        trans: [{to: name+"-I", weight: "gapExtend"},
			        {to: name+"-W", weight: not("gapExtend")}]},
		       {id: name+"-W",
		        trans: [{to: name+"-M", weight: not("gapOpen")},
			        {to: name+"-D", weight: "gapOpen"}]}]
                    : [])
                   .concat ([{id: name+"-M",
		              trans: [{to: name+"-E"}]
		              .concat (alph.reduce ((tc,c) => tc.concat ((pswFlag
                                                                          ? alph.map ((d) => { return { in: c, out: d, to: startState, weight: times(not("intron"),"sub"+c+d) } })
                                                                          : [{ in: c, out: c, to: startState, weight: not("intron") }])
								         .concat ([{ in: c, to: name+"-M-intron", weight: "intron" }])), []))}])
                   .concat (pswFlag
                            ? [{id: name+"-D",
		                trans: [{to: name+"-E"}]
		                .concat (alph.map ((c) => {return { in: c, to: name+"-X" }}))},
		               {id: name+"-X",
		                trans: [{to: name+"-D", weight: "gapExtend"},
			                {to: name+"-M", weight: not("gapExtend")}]}]
                            : [])
	           .concat (intronStates("M"))
	           .concat (pswFlag ? intronStates("I") : [])
	           .concat ([{id: name+"-E"}])),
           cons: cons }
}

module.exports = {
  alph: alph,
  machine: machine
}
