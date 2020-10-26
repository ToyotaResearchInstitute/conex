#include "conex/divergence.h"

#include <cmath>


#include "conex/debug_macros.h"

namespace {

// Finds the largest (when a > 0) solution to 
//
//  (x^2 a + b x + c) / (2 - d x) = k.
//
// Formula found using Wolfram-Alpha code with input:
//      
//      solve ((a*x^2 + b*x + c)/(2-x*d) -k, x)
double SolveRationalEquation(double a, double b, double c, double d, double k) {
  double under_radical = b*b - 4 * a * c + 8 * a * k + 2 * b * d * k + std::pow(d* k, 2);
  double x1 = -(b + d*k + std::sqrt(under_radical))/(2 * a);
  double x2 = -(b + d*k - std::sqrt(under_radical))/(2 * a);
  if (d*x2 > 0) {
    return x2;
  } else {
    return x1;
  }
}
}

double DivergenceUpperBoundInverse(double divergence_upper_bound, 
                                   double gw_norm_squared, 
                                   double gw_norm_inf, 
                                   double gw_trace, 
                                   int rank) {
  double a = gw_norm_squared;
  double b = -2 * gw_trace;
  double c = rank;
  double d = gw_norm_inf;

  double x =  SolveRationalEquation(a, b, c, d, divergence_upper_bound);
  return x;
}


double DivergenceUpperBound(double k, double gw_norm_squared, double gw_norm_inf, double gw_trace, int n) {
  double numerator = k*k * gw_norm_squared  - 2 * k * gw_trace + n;
  double denom = 2 - k * gw_norm_inf;
  return numerator / denom;
}

