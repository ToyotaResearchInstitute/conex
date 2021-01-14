#include "gtest/gtest.h"
#include <Eigen/Dense>

#include "conex/cone_program.h"
#include "conex/dense_lmi_constraint.h"
#include "conex/linear_constraint.h"
#include "conex/test/test_util.h"

namespace conex {

using Eigen::MatrixXd;

GTEST_TEST(Warmstart, AgreesWithFullSolveIfNoDataIsChanged) {
  SolverConfiguration config;
  config.inv_sqrt_mu_max = 10000000;
  config.final_centering_steps = 0;
  int num_iters = 10;
  int n = 15;
  int m = 13;
  auto constraints2 = GetRandomDenseMatrices(n, m);

  DenseMatrix affine2 = Eigen::MatrixXd::Identity(n, n);
  DenseLMIConstraint LMI{n, constraints2, affine2};

  Program prog(m);
  DenseMatrix y(m, 1);
  prog.AddConstraint(LMI);

  auto b = GetFeasibleObjective(&prog);
  config.max_iterations = num_iters;
  Solve(b, prog, config, y.data());

  DenseMatrix ywarm(m, 1);
  for (int i = 0; i < num_iters; i++) {
    config.initialization_mode = i != 0;
    config.max_iterations = 1;
    config.final_centering_steps = 0;
    Solve(b, prog, config, ywarm.data());
  }
  EXPECT_NEAR((y - ywarm).norm(), 0, 1e-12);
}
/*
GTEST_TEST(Warmstart, ObjectivePertubation) {
  SolverConfiguration config;
  config.inv_sqrt_mu_max = 1000;
  config.final_centering_steps = 0;
  config.divergence_upper_bound = 10000;
  int num_iters = 20;
  int n = 15;
  int m = 13;
  auto constraints2 = GetRandomDenseMatrices(n, m);

  DenseMatrix affine2 = Eigen::MatrixXd::Identity(n, n);
  DenseLMIConstraint LMI{n, constraints2, affine2};

  Program prog;
  DenseMatrix y(m, 1);
  prog.AddConstraint(LMI);

  auto b = GetFeasibleObjective(m, prog.constraints);
  config.max_iterations = num_iters;
  Solve(b, prog, config, y.data());

  b = b + MatrixXd::Random(m, 1) * .05;
  DenseMatrix ywarm(m, 1);
  config.initialization_mode = 1;
  Solve(b, prog, config, ywarm.data());
}*/

}  // namespace conex
