#!/usr/bin/env node

var prot = "ACDEFGHIKLMNPQRSTVWY".split("")

var machine = { state: [{ id: "S",
                          trans: prot.reduce ((t,c) => [
                            { in: c, out: c, to: "S", weight: {"-":[true,"intron"]} },
                            { in: c, to: "IBB", weight: {"/":["intron",3]} },
                            { in: c, to: "BIB", weight: {"/":["intron",3]} },
                            { in: c, to: "BBI", weight: {"/":["intron",3]} }
                          ]) },
                        { id: "IBB", trans: [{ out: "intron", to: "BB" }] },
                        { id: "BB", trans: [{ out: "base", to: "B" }] },
                        { id: "B", trans: [{ out: "base", to: "S" }] },
                        { id: "BIB", trans: [{ out: "base", to: "IB" }] },
                        { id: "IB", trans: [{ out: "intron", to: "B" }] },
                        { id: "BBI", trans: [{ out: "base", to: "BI" }] },
                        { id: "BI", trans: [{ out: "base", to: "I" }] },
                        { id: "I", trans: [{ out: "intron", to: "S" }] }] }

console.log (JSON.stringify (machine))
