var jStat = require('jStat').jStat

// test as follows
// node -e 'negbin=require("./negbin.js");dist=[0].concat(process.argv.slice(2).map((x)=>parseInt(x)));k=parseInt(process.argv[1]);result=negbin.fitNegBin(dist,k,undefined,true);console.log(result)' 1 32 16 8 4 2 1

module.exports = { logProbNegBin: logProbNegBin,
                   fitNegBin: fitNegBin,
                   minFracInc: minFracInc }

function sumArray (list) {
  return list.reduce (function (sum, x) { return sum + x }, 0)
}

function logProbNegBin (lenDist, rDist, pExtend) {
  var lp = 0
  lenDist.forEach (function (count, len) {
    if (count) {
      var lenProb = 0
      rDist.forEach (function (rProb, r) {
        if (rProb && r <= len) {
          lenProb += rProb * jStat.negbin.pdf (len - r, r, 1 - pExtend)
//          console.log("len="+len+" r="+r+" rProb="+rProb+" negbin="+jStat.negbin.pdf (len - r, r, 1 - pExtend))
        }
      })
//      console.log("len="+len+" lenProb="+lenProb)
      lp += Math.log(lenProb) * count
    }
  })
  return lp
}

function fitNegBin (lenDist, rDist, pExtend, callback) {
  var lpPrev
  if (typeof(rDist) === 'number')
    rDist = [0].concat (new Array(rDist).fill(1/rDist))
  if (!pExtend)
    pExtend = 1/2
  if (rDist[0])
    throw new Error ("Component weight of r=0 in negative binomial mixture must be zero")
  if (lenDist[0])
    throw new Error ("Observed frequency of n=0 when fitting negative binomial mixture must be zero")
  if (callback && typeof(callback) !== 'function')
    callback = function (stats) {
      console.warn ("Iteration " + stats.iteration + ": log-likelihood=" + stats.logLike + " P(extend)=" + stats.pExtend + " P(r)=[" + stats.rDist.join(",") + "] n(extend,end)=[" + stats.extendCount + "," + stats.endCount + "] n(r)=[" + stats.rCount.join(",") + "]")
    }
  for (var iter = 1; true; ++iter) {
    var lp = logProbNegBin (lenDist, rDist, pExtend)
    var rCount = rDist.map (function() { return 0 })
    var extendCount = 0, endCount = 0
    lenDist.forEach (function (count, len) {
      if (count) {
        var rLike = rDist.map (function (rProb, r) {
          return (r > len || !rProb) ? 0 : rProb * jStat.negbin.pdf (len - r, r, 1 - pExtend)
        })
        var lenProb = sumArray (rLike)
        rLike.forEach (function (like, r) {
          var rPostProb = like / lenProb
          rCount[r] += count * rPostProb
          endCount += count * rPostProb * r
          extendCount += count * rPostProb * (len - r)
        })
      }
    })
    if (callback && callback ({ iteration: iter,
                                logLike: lp,
                                prevLogLike: lpPrev,
                                rDist: rDist,
                                pExtend: pExtend,
                                pEnd: 1 - pExtend,
                                rCount: rCount,
                                extendCount: extendCount,
                                endCount: endCount }))
      break
    if (typeof(lpPrev) !== 'undefined' && lp <= lpPrev)
      break
    lpPrev = lp
    var rCountSum = sumArray (rCount)
    rCount.forEach (function (count, r) { rDist[r] = count / rCountSum })
    pExtend = extendCount / (extendCount + endCount)
  }
  return { rDist: rDist, pExtend: pExtend, pEnd: 1 - pExtend, logLike: lpPrev }
}

function minFracInc (mfi) {
  return function (stats) {
    return stats.iteration > 1
      && (stats.logLike - stats.prevLogLike) / Math.abs(stats.prevLogLike) < mfi
  }
}
