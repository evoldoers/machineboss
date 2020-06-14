#!/usr/bin/env node

const aa = 'ACDEFGHIKLMNPQRSTVWY'.split('')

const machine = { state: [{ n: 0,
                            trans: aa.map((c) => { return { to: 0, in: c, out: c } })
                            .concat (aa.map((c) => { return { to: 0, in: 'X', out: c } })) }] }

console.log (JSON.stringify (machine))
