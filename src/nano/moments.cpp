#include <math.h>
#include <gsl/gsl_math.h>
#include "../util.h"
#include "moments.h"
#include "segmenter.h"

#include "../../ext/fast5/fast5.hpp"

SampleMoments::SampleMoments() :
  m0(0), m1(0), m2(0)
{ }

SampleMoments::SampleMoments (const Trace& t, size_t pos, size_t len) :
  m0(0), m1(0), m2(0)
{
  for (size_t i = 0; i < len; ++i) {
    const double x = t.sample[pos + i];
    m0 += 1;
    m1 += x;
    m2 += x*x;
  }
}

TraceMoments::TraceMoments()
{ }

TraceMoments::TraceMoments (const Trace& trace) :
  name (trace.name),
  sample (trace.sample.size())
{
  for (size_t j = 0; j < sample.size(); ++j) {
    const double t = trace.sample[j];
    SampleMoments& e = sample[j];
    e.m0 = 1;
    e.m1 = t;
    e.m2 = t*t;
  }
}

void TraceMoments::readFast5 (const string& filename) {
  name = filename;
  sample.clear();

  fast5::File f;
  f.open(filename);
  const vector<fast5::EventDetection_Event> events = f.get_eventdetection_events();
  sample.resize (events.size());
  for (size_t n = 0; n < events.size(); ++n) {
    const fast5::EventDetection_Event& e = events[n];
    SampleMoments& s = sample[n];
    s.m0 = e.length;
    s.m1 = e.mean * e.length;
    s.m2 = (e.stdv * e.stdv + e.mean * e.mean) * e.length;
  }
}

void TraceMoments::assertIsSummaryOf (const Trace& trace, double epsilon) const {
  size_t pos = 0;
  for (const auto& x: sample) {
    Assert (pos + (size_t) x.m0 <= trace.sample.size(), "Trace summary %s has more samples than original trace %s", name.c_str(), trace.name.c_str());
    const SampleMoments s (trace, pos, x.m0);
    Assert (gsl_fcmp(s.m1,x.m1,epsilon) == 0, "Samples %u-%u of trace summary %s have 1st moment %g, original trace %s has 1st moment %g (difference %g)", pos, pos + x.m0 - 1, name.c_str(), x.m1, trace.name.c_str(), s.m1, abs(s.m1-x.m1));
    Assert (gsl_fcmp(s.m2,x.m2,epsilon) == 0, "Samples %u-%u of trace summary %s have 2nd moment %g, original trace %s has 2nd moment %g (difference %g)", pos, pos + x.m0 - 1, name.c_str(), x.m2, trace.name.c_str(), s.m2, abs(s.m2-x.m2));
    pos += (size_t) x.m0;
  }
  Assert (pos == trace.sample.size(), "Trace summary %s has fewer samples than original trace %s", name.c_str(), trace.name.c_str());
}

void TraceMoments::writeJson (ostream& out) const {
  out << "{\"name\":\"" << escaped_str(name) << "\",\"moments\":[";
  for (size_t n = 0; n < sample.size(); ++n)
    out << (n ? "," : "") << "[" << sample[n].m0 << "," << sample[n].m1 << "," << sample[n].m2 << "]";
  out << "]}";
}

string TraceMoments::pathScoreBreakdown (const Machine& machine, const MachinePath& path, const GaussianModelParams& modelParams, const TraceParams& traceParams) const {
  ostringstream out;
  //  out << "Source\tDest\tIn\tOut\tTrans\tEmit\n";
  out << "{ \"trans\":\n  [";
  const OutputTokenizer outputTokenizer (machine.outputAlphabet());
  const GaussianModelCoefficients modelCoeffs (modelParams, traceParams, outputTokenizer);
  const auto params = modelParams.params (traceParams.rate);
  StateIndex s = machine.startState();
  long pos = 0, ignore = 0, transCount = 0;
  double lp = 0;
  for (const auto& trans: path.trans) {
    const double trans_ll = log (WeightAlgebra::eval (trans.weight, params.defs));
    double emit_ll = 0;
    out << ((transCount++) ? ",\n   " : "")
	<< "{ \"src\": " << machine.stateNameJson(s)
	<< ", \"dest\": " << machine.stateNameJson(trans.dest);
    if (!trans.inputEmpty())
      out << ", \"in\": \"" << trans.in << "\"";
    if (!trans.outputEmpty())
      out << ", \"out\": \"" << trans.out << "\"";
    out << ", \"transLogLike\": " << trans_ll;
    if (!trans.outputEmpty()) {
      if (ignore > 0) {
	--ignore;
	out << ", \"segment\": true";
      } else {
	const SampleMoments& x = sample[pos++];
	const GaussianCoefficients& gc = modelCoeffs.gauss[outputTokenizer.sym2tok.at (trans.out) - 1];
	ignore = x.m0 - 1;
	emit_ll = gc.logEmitProb(x);
	out << ", \"emitLogLike\": " << emit_ll;
      }
      lp += trans_ll + emit_ll;
    }
    out << " }";
    s = trans.dest;
  }
  out << "]," << endl
      << " \"logLike\": " << lp
      << " }" << endl;
  return out.str();
}

TraceMomentsList::TraceMomentsList()
{ }

TraceMomentsList::TraceMomentsList (const TraceList& traceList) {
  init (traceList);
}

TraceMomentsList::TraceMomentsList (const TraceList& traceList, double maxFracDiff, size_t maxSegLen) {
  init (traceList, maxFracDiff, maxSegLen);
}

void TraceMomentsList::init (const TraceList& traceList) {
  for (const auto& t: traceList.trace)
    trace.push_back (TraceMoments(t));
}

void TraceMomentsList::init (const TraceList& traceList, double maxFracDiff, size_t maxSegLen) {
  for (const auto& t: traceList.trace) {
    const Segmenter seg (t, maxFracDiff, maxSegLen);
    trace.push_back (seg.segments());
  }
}

void TraceMomentsList::readFast5 (const string& filename) {
  TraceMoments m;
  m.readFast5 (filename);
  trace.push_back (m);
}

void TraceMomentsList::assertIsSummaryOf (const TraceList& traceList, double epsilon) const {
  auto iter = trace.begin();
  auto tlIter = traceList.trace.begin();
  while (iter != trace.end()) {
    Assert (tlIter != traceList.trace.end(), "More traces in trace moments list than in original trace list");
    (*iter).assertIsSummaryOf (*tlIter, epsilon);
    ++tlIter;
    ++iter;
  }
  Assert (tlIter == traceList.trace.end(), "Fewer traces in trace moments list than in original trace list");
}

ostream& operator<< (ostream& out, const TraceMomentsList& traceMomentsList) {
  size_t n = 0;
  out << "[";
  for (const auto& trace: traceMomentsList.trace) {
    if (n++)
      out << ",\n ";
    trace.writeJson (out);
  }
  out << "]";
  return out;
}

GaussianModelCoefficients::GaussianModelCoefficients (const GaussianModelParams& modelParams, const TraceParams& traceParams, const OutputTokenizer& outputTokenizer)
  : gauss (outputTokenizer.tok2sym.size() - 1)
{
  const double log_sqrt_2pi = log(2*M_PI)/2;
  const double shift = traceParams.shift, scale = traceParams.scale;
  for (OutputToken outTok = 1; outTok < outputTokenizer.tok2sym.size(); ++outTok) {
    const auto& outSym = outputTokenizer.tok2sym[outTok];
    const GaussianParams& p = modelParams.gauss.at(outSym);
    GaussianCoefficients& e = gauss[outTok-1];
    const double mu_plus_shift = p.mu + shift;
    e.m0coeff = log(p.tau)/2 - log(scale) - log_sqrt_2pi - (p.tau/2)*mu_plus_shift*mu_plus_shift;
    e.m1coeff = (p.tau/scale) * mu_plus_shift;
    e.m2coeff = -p.tau/(2*scale*scale);
  }
}
