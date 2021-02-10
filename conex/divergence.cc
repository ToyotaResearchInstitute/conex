#include "conex/divergence.h"

#include <cmath>

#include "conex/debug_macros.h"

namespace conex {

namespace {

// Finds the largest (when a > 0) solution to
//
//  (x^2 a + b x + c) / (2 - d x) = k.
//
// Formula found using Wolfram-Alpha code with input:
//
//      solve ((a*x^2 + b*x + c)/(2-x*d) -k, x)
double SolveRationalEquation(double a, double b, double c, double d, double k) {
  double under_radical =
      b * b - 4 * a * c + 8 * a * k + 2 * b * d * k + std::pow(d * k, 2);
  double x2 = -(b + d * k - std::sqrt(under_radical)) / (2 * a);
  return x2;
}
}  // namespace

double InverseLambdaMaxBranch(double divergence_upper_bound,
                              const MuSelectionParameters& p) {
  double a = p.gw_norm_squared;
  double b = -2 * p.gw_trace;
  double c = p.rank;
  double d = p.gw_lambda_max;

  double x = SolveRationalEquation(a, b, c, d, divergence_upper_bound);
  double lower_bound = 2.0 / (p.gw_lambda_max + p.gw_lambda_min);

  double k = -1;
  if (x >= lower_bound) {
    k = x;
  }
  return k;
}

bool InLimits(double x, double lower, double upper) {
  return x >= lower && x <= upper;
}

// a k - b + n/k = c
bool SolveQuadratic(double a, double b, double n, double c,
                    std::pair<double, double>* sol) {
  double under_radical = b * b + 2 * b * c + c * c - 4 * a * n;
  std::pair<double, double> solution;
  sol->first = (b + c + std::sqrt(under_radical)) / (2 * a);
  sol->second = (b + c - std::sqrt(under_radical)) / (2 * a);

  if (under_radical < 0) {
    return false;
  } else {
    return true;
  }
}

// TODO(FrankPermenter): Analyze if both solutions of Quadratic
// are needed.
double InverseLambdaMinBranch(double divergence_upper_bound,
                              const MuSelectionParameters& p) {
  double lower_bound = 0;
  double upper_bound = 2.0 / (p.gw_lambda_max + p.gw_lambda_min);
  double k = -1;
  std::pair<double, double> k2;
  if (SolveQuadratic(p.gw_norm_squared / p.gw_lambda_min,
                     2 * p.gw_trace / p.gw_lambda_min, p.rank / p.gw_lambda_min,
                     divergence_upper_bound, &k2)) {
    if (InLimits(k2.first, lower_bound, upper_bound)) {
      k = k2.first;
    }
    if (InLimits(k2.second, lower_bound, upper_bound)) {
      if (k2.second > k) {
        k = k2.second;
      }
    }
  }
  return k;
}

bool BoundIsFinite(double k, MuSelectionParameters& p) {
  double norm_inf = std::fabs(k * p.gw_lambda_max - 1);
  if (norm_inf < std::fabs(k * p.gw_lambda_min - 1)) {
    norm_inf = std::fabs(k * p.gw_lambda_min - 1);
  }
  if (norm_inf < 1) {
    return 1;
  }
  return 0;
}

double DivergenceUpperBoundInverse(double divergence_upper_bound,
                                   MuSelectionParameters& p) {
  double k = -1;
  double k1 = InverseLambdaMinBranch(divergence_upper_bound, p);
  double k2 = InverseLambdaMaxBranch(divergence_upper_bound, p);

  if (BoundIsFinite(k1, p)) {
    k = k1;
  }

  if (k2 > k && BoundIsFinite(k2, p)) {
    k = k2;
  }

  return k;
}

double DivergenceUpperBound(double k, MuSelectionParameters& p) {
  double numerator = k * k * p.gw_norm_squared - 2 * k * p.gw_trace + p.rank;
  double norm_inf = std::fabs(k * p.gw_lambda_max - 1);
  if (norm_inf < std::fabs(k * p.gw_lambda_min - 1)) {
    norm_inf = std::fabs(k * p.gw_lambda_min - 1);
  }
  return numerator / (1 - norm_inf);
}

}  // namespace conex
