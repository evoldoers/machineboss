#include "obj_gaussian.h"

GaussianOutputAdapter::GaussianOutputAdapter()
  : meanParam("mu"),
    sdParam("sigma"),
    shiftParam("shift"),
    scaleParam("scale"),
    m0Param("moment0"),
    m1Param("moment1"),
    m2Param("moment2")
{
  symParams.push_back (meanParam);
  symParams.push_back (sdParam);
  seqParams.push_back (shiftParam);
  seqParams.push_back (scaleParam);
  
  // log(P(x)) = A + B*x + C*x^2
  // where...
  //  A = (mu+shift)^2/(2*sigma^2) - log(scale*sigma) - log(2*pi)/2
  //  B = (mu+shift)/(scale*sigma^2)
  //  C = 1/(2*(scale*sigma)^2)
  // We drop the -log(2*pi)/2 term from A as it doesn't depend on any parameters
  auto add = WeightAlgebra::add, multiply = WeightAlgebra::multiply, subtract = WeightAlgebra::subtract, divide = WeightAlgebra::divide;
  auto logOf = WeightAlgebra::logOf;
  auto mu_plus_shift = add (meanParam, shiftParam);
  auto scale_times_sigma = multiply (scaleParam, sdParam);
  auto sigma_squared = multiply (sdParam, sdParam);
  summaryStatCoeff[m0Param] = subtract (divide (multiply(mu_plus_shift,mu_plus_shift),
						multiply(2,sigma_squared)),
					logOf (scale_times_sigma));  // A
  summaryStatCoeff[m1Param] = divide (mu_plus_shift, multiply (scaleParam, sigma_squared));  // B
  summaryStatCoeff[m2Param] = divide (1, multiply (2, multiply (scale_times_sigma, scale_times_sigma)));  // C
}

OutputAdapter::SummaryStats GaussianOutputAdapter::summaryStats (const OutputObject& outputUnit) const {
  SummaryStats stats;
  const double x = outputUnit.get<double>();
  stats[m0Param] = 1;
  stats[m1Param] = x;
  stats[m2Param] = x*x;
  return stats;
}
