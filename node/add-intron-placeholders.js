#!/usr/bin/env node

var prot = "ACDEFGHIKLMNPQRSTVWY".split("")

var machine = { state: [{ id: "i-S",
                          trans: prot.reduce ((t,c) => t.concat([
                            { to: "i-E" },
                            { in: c, out: c, to: "i-S", weight: {"-":[true,"intron"]} },
                            { in: c, to: "i-IBB", weight: {"/":["intron",3]} },
                            { in: c, to: "i-BIB", weight: {"/":["intron",3]} },
                            { in: c, to: "i-BBI", weight: {"/":["intron",3]} }
                          ]), []) },
                        { id: "i-IBB", trans: [{ out: "intron", to: "i-BB" }] },
                        { id: "i-BB", trans: [{ out: "base", to: "i-B" }] },
                        { id: "i-B", trans: [{ out: "base", to: "i-S" }] },
                        { id: "i-BIB", trans: [{ out: "base", to: "i-IB" }] },
                        { id: "i-IB", trans: [{ out: "intron", to: "i-B" }] },
                        { id: "i-BBI", trans: [{ out: "base", to: "i-BI" }] },
                        { id: "i-BI", trans: [{ out: "base", to: "i-I" }] },
                        { id: "i-I", trans: [{ out: "intron", to: "i-S" }] },
			{ id: "i-E" }] }

console.log (JSON.stringify (machine))
