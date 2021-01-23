#include <cmath>
#include "conex/debug_macros.h"
#include "conex/divergence.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

namespace conex {

using Eigen::MatrixXd;

bool InLimits(double x, double lower, double upper) {
  return x >= lower && x <= upper;
}

double Divergence(Eigen::VectorXd gw, double k) {
  MatrixXd d = k * gw;
  d.array() -= 1;
  double dinf = d.array().abs().maxCoeff();
  return d.squaredNorm() / (1 - dinf);
}

GTEST_TEST(MuSelection, DivergenceBound) {
  int n = 3;
  MatrixXd gw = Eigen::VectorXd::Random(n, 1).array().abs();

  MuSelectionParameters p;
  p.gw_norm_squared = gw.squaredNorm();
  p.gw_lambda_max = gw.maxCoeff();
  p.gw_lambda_min = gw.minCoeff();
  p.gw_trace = gw.sum();
  p.rank = gw.rows();

  // Div bound on one branch.
  double k_ref = 2.0 / (p.gw_lambda_max + p.gw_lambda_min) * .8;
  double hub_desired = Divergence(gw, k_ref);
  EXPECT_NEAR(hub_desired, DivergenceUpperBound(k_ref, p), 1e-8);
  double k = DivergenceUpperBoundInverse(hub_desired, p);
  EXPECT_NEAR(hub_desired, Divergence(gw, k), 1e-8);
  EXPECT_TRUE(k >= k_ref);

  // Div bound on the other branch.
  k_ref = 2.0 / (p.gw_lambda_max + p.gw_lambda_min) * 1.2;
  hub_desired = Divergence(gw, k_ref);
  EXPECT_NEAR(hub_desired, DivergenceUpperBound(k_ref, p), 1e-8);
  k = DivergenceUpperBoundInverse(hub_desired, p);
  EXPECT_NEAR(hub_desired, DivergenceUpperBound(k, p), 1e-8);
  EXPECT_TRUE(k >= k_ref);

  // Div bound undefined.
  k_ref = 1000000;
  hub_desired = Divergence(gw, k_ref);
  EXPECT_NEAR(hub_desired, DivergenceUpperBound(k_ref, p), 1e-8);
  k = DivergenceUpperBoundInverse(hub_desired, p);
  EXPECT_TRUE(DivergenceUpperBound(k, p) < 0);
  EXPECT_EQ(k, -1);
}

}  // namespace conex
