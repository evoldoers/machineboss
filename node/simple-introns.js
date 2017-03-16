#!/usr/bin/env node

var prot = "ACDEFGHIKLMNPQRSTVWY".split("")

var machine = { state: [{ id: "S",
                          trans: prot.map ((c) => { return { in: c, out: c, to: "S" } })
                          .concat ([{ in: "base", out: "base", to: "S"},
                                    { in: "intron", out: "G", to: "donor" },
				    { to: "E" }]) },
                        { id: "donor", trans: [{ out: "T", to: "intron" }] },
                        { id: "intron", trans: [{ out: "base", to: "intron", weight: "extendIntron" },
                                                { out: "A", to: "acceptor", weight: {"-":[true,"extendIntron"]} }] },
                        { id: "acceptor", trans: [{ out: "G", to: "S" }] },
			{ id: "E" }] }

console.log (JSON.stringify (machine))
