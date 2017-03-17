#!/usr/bin/env node

var prot = "ACGT".split("")

var machine = { state: [{ id: "si-S",
                          trans: prot.map ((c) => { return { in: c, out: c, to: "si-S" } })
                          .concat ([{ in: "base", out: "base", to: "si-S"},
                                    { in: "intron", out: "G", to: "si-donor" },
				    { to: "si-E" }]) },
                        { id: "si-donor", trans: [{ out: "T", to: "si-intron" }] },
                        { id: "si-intron", trans: [{ out: "base", to: "si-intron", weight: "extendIntron" },
                                                { out: "A", to: "si-acceptor", weight: {"-":[true,"extendIntron"]} }] },
                        { id: "si-acceptor", trans: [{ out: "G", to: "si-S" }] },
			{ id: "si-E" }] }

console.log (JSON.stringify (machine))
