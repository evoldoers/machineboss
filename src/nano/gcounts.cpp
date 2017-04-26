#include "gcounts.h"

GaussianCounts::GaussianCounts() : m0(0), m1(0), m2(0)
{ }

void GaussianCounts::inc (const SampleMoments& sampleMoments, const double postProb) {
  m0 += sampleMoments.m0 * postProb;
  m1 += sampleMoments.m1 * postProb;
  m2 += sampleMoments.m2 * postProb;
}
