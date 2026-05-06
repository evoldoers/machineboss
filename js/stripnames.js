#!/usr/bin/env node

var fs = require('fs'),
    exec = require('child_process').exec

var command = process.argv.slice(2).join(" ")
console.warn (command)
exec (command, function (error, stdout, stderr) {
  var json = JSON.parse (stdout)
  var rounded = json.map (function (tuple) {
    var v = tuple[2]
    if (typeof v === 'number' && isFinite(v))
      v = Number (Number(v).toPrecision(6))
    return [v]
  })
  console.log (JSON.stringify (rounded))
})
