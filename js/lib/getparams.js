function getParams (params, names, p) {
  var ok = true
  for (var n = 0; n < names.length; ++n) {
    const name = names[n]
    if (!params.hasOwnProperty (name)) {
      console.warn ("Please define parameter: " + name)
      ok = false
    } else
      p[n] = params[name]
  }
  return ok
}

module.exports = getParams
