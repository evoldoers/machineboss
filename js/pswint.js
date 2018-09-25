#!/usr/bin/env node

var machine = require ('./lib/pswint').machine
console.log (JSON.stringify (machine (true)))
