const boss = require ('./boss')

const initPromise = new Promise ((resolve, reject) => {
  boss.onRuntimeInitialized = resolve
})

const wrapBoss = (args) => {
  return initPromise
    .then (() => {
      args = args || [];

      var nFiles = 0, filePrefix = 'FILE'
      const wrappedArgs = args.map ((opt) => {
	if (typeof(opt) !== 'string') {
	  const fileBuffer = JSON.stringify (opt)
	  const filename = filePrefix + (++nFiles)
	  boss.FS.writeFile (filename, new Uint8Array(fileBuffer))
	  return filename
	} else
	  return opt
      })

      boss.stdout = []
      boss.callMain (wrappedArgs)
      const result = JSON.parse (boss.stdout.join(''))
      delete boss.stdout
      return result
    })
}

module.exports = { initialized: initPromise,
		   run: wrapBoss,
		   boss: boss }
