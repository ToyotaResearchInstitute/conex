#include <cmath>

namespace {

// Finds the largest (when a > 0) solution to 
//
//  (x^2 a + b x + c) / (2-x) = k.
//
// Formula found using Wolfram-Alpha code with input:
//      
//      solve ((a*x^2 + b*x + c)/(2-x*d) -k, x)
double SolveRationalEquation(double a, double b, double c, double d, double k) {
  double under_radical = b*b - 4 * a * c + 8 * a * k + 2 * b * d * k + std::pow(d* k, 2);
  // double y = -(b + d*k + std::sqrt(under_radical))/(2 * a);
  return -(b + d*k - std::sqrt(under_radical))/(2 * a);
}
}

double DivergenceUpperBoundInverse(double divergence_upper_bound, 
                                   double gw_norm, 
                                   double gw_norm_inf, 
                                   double gw_trace, 
                                   int rank) {
  double a = gw_norm * gw_norm;
  double b = -2 * gw_trace;
  double c = rank;
  double d = gw_norm_inf;

  return SolveRationalEquation(a, b, c, d, divergence_upper_bound);
}


double DivergenceBound(double k, double gw_norm, double gw_norm_inf, double gw_trace, int n) {
  double numerator = k*k * gw_norm * gw_norm  - 2 * k * gw_trace + n;
  double denom = 2 - k * gw_norm_inf;
  return numerator / denom;
}

