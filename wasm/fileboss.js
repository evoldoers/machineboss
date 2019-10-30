#!/usr/bin/env node

var opts = process.argv.slice(2)

var boss = require ('./boss.js')
var fs = require ('fs')

boss.onRuntimeInitialized = () => {
  var nFiles = 0, filePrefix = 'FILE'
  opts = opts.map ((opt) => {
    if (opt.charAt(0) !== '-'
	&& fs.existsSync(opt)
	&& fs.lstatSync(opt).isFile()) {
      const fileBuffer = fs.readFileSync(opt)
      const filename = filePrefix + (++nFiles)
      boss.FS.writeFile (filename, new Uint8Array(fileBuffer))
      return filename
    } else
      return opt
  })

  boss.callMain (opts)
}
