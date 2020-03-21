const boss = require ('./boss')

const initPromise = new Promise ((resolve, reject) => {
  boss.onRuntimeInitialized = resolve
})

const wrapBoss = (args) => {
  return initPromise
    .then (() => {
      args = args || [];

      let nFiles = 0, filePrefix = 'FILE'
      const wrappedArgs = args.map ((opt) => {
	if (typeof(opt) !== 'string') {
	  const fileBuffer = Buffer.from (JSON.stringify (opt), 'utf-8')
	  const filename = filePrefix + (++nFiles)
	  boss.FS.writeFile (filename, new Uint8Array(fileBuffer))
	  return filename
	} else
	  return opt
      })

      boss.stdout = []
      boss.callMain (wrappedArgs)
      let result, output = boss.stdout.join('')
      try {
        result = JSON.parse (output)
      } catch (e) {
        result = output
      }
      delete boss.stdout
      return result
    })
}

module.exports = { initialized: initPromise,
		   run: wrapBoss,
		   boss: boss }
