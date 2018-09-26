#!/usr/bin/env node

var fs = require('fs'),
    exec = require('child_process').exec

var command = process.argv.slice(2).join(" ")
console.warn (command)
exec (command, function (error, stdout, stderr) {
  var json = JSON.parse (stdout)
  console.log (JSON.stringify (json.map (function (tuple) { return [tuple[2]] })))
})
