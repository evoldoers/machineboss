
const SOFTPLUS_CACHE_MAX_LOG    = 10;
const SOFTPLUS_INTLOG_PRECISION = .0001;
const SOFTPLUS_CACHE_ENTRIES    = Math.floor ((SOFTPLUS_CACHE_MAX_LOG / SOFTPLUS_INTLOG_PRECISION) + 1);

const SOFTPLUS_INTLOG_INFINITY = 0x1FFFFFFFFFFFFFFF;
const SOFTPLUS_LOG_INFINITY    = (SOFTPLUS_INTLOG_PRECISION * SOFTPLUS_INTLOG_INFINITY);

var cache = new Array (SOFTPLUS_CACHE_ENTRIES);
for (var n = 0; n < SOFTPLUS_CACHE_ENTRIES; ++n)
  cache[n] = log_to_int (softplus (-int_to_log (n)));

function softplus (x) {
  return Math.log (1 + Math.exp (x));
}

function int_softplus_neg (x) {
  if (x < 0)
    throw new Error ("int_softplus_neg: negative argument");
  return x >= SOFTPLUS_CACHE_ENTRIES ? 0 : cache[x|0];
}

function log_to_int (x) {
  return (x <= -SOFTPLUS_LOG_INFINITY
	    ? -SOFTPLUS_INTLOG_INFINITY
	    : (x >= SOFTPLUS_LOG_INFINITY
	       ? SOFTPLUS_INTLOG_INFINITY
	       : (Math.floor (.5 + x / SOFTPLUS_INTLOG_PRECISION))));
}


function int_logsumexp_canonical (larger, smaller) {
  return (smaller <= -SOFTPLUS_INTLOG_INFINITY || larger >= SOFTPLUS_INTLOG_INFINITY
	  ? bound_intlog (larger)
	  : (larger + int_softplus_neg (larger - smaller)));
}

function int_log (x) {
  return (x > 0
	  ? log_to_int (Math.log (x))
	  : -SOFTPLUS_INTLOG_INFINITY);
}

function int_exp (x) {
  return Math.exp (int_to_log (x));
}

function int_logsumexp (a, b) {
  return (a > b
	  ? int_logsumexp_canonical (a, b)
	  : int_logsumexp_canonical (b, a));
}

function bound_intlog (x) {
  return (x < -SOFTPLUS_INTLOG_INFINITY
	  ? -SOFTPLUS_INTLOG_INFINITY
	  : (x > SOFTPLUS_INTLOG_INFINITY
	     ? SOFTPLUS_INTLOG_INFINITY
	     : x));
}

function int_to_log (x) {
  return (x <= -SOFTPLUS_INTLOG_INFINITY
	  ? -Infinity
	  : (x >= SOFTPLUS_INTLOG_INFINITY
	     ? Infinity
	     : (SOFTPLUS_INTLOG_PRECISION * x)));
}

module.exports = {
  int_log: int_log,
  int_exp: int_exp,
  int_logsumexp: int_logsumexp,
  int_to_log: int_to_log,
  bound_intlog: bound_intlog,
  SOFTPLUS_INTLOG_INFINITY: SOFTPLUS_INTLOG_INFINITY,
  SOFTPLUS_LOG_INFINITY: SOFTPLUS_LOG_INFINITY
}
