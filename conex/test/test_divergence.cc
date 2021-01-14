#include <cmath>
#include "conex/debug_macros.h"
#include "conex/divergence.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

namespace conex {

using Eigen::MatrixXd;

GTEST_TEST(MuSelection, DivergenceBound) {
  int n = 3;
  MatrixXd gw = Eigen::VectorXd::Random(n, 1).array().abs();

  double gw_norm_squared = gw.squaredNorm();
  double gw_norm_inf = gw.maxCoeff();
  double gw_trace = gw.sum();
  double hub_desired = 1;
  double k = DivergenceUpperBoundInverse(hub_desired, gw_norm_squared,
                                         gw_norm_inf, gw_trace, n);
  double hub =
      DivergenceUpperBound(k, gw_norm_squared, gw_norm_inf, gw_trace, n);

  EXPECT_TRUE(k >= 0);
  EXPECT_TRUE(2 - k * gw_norm_inf >= 0);
  EXPECT_NEAR(hub, hub_desired, 1e-12);
}

}  // namespace conex
