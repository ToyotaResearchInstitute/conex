#include "conex/cone_program.h"
#include "conex/equality_constraint.h"
#include "conex/linear_constraint.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

namespace conex {

using Eigen::MatrixXd;
using Eigen::VectorXd;
GTEST_TEST(EqualityConstraints, Basic) {
  int num_vars = 3;
  int num_equalities = 1;
  int num_inequalities = 4;

  MatrixXd A = MatrixXd::Random(num_inequalities, num_vars);
  MatrixXd C(num_inequalities, 1);
  MatrixXd b(num_vars, 1);

  MatrixXd optimal_slack(num_inequalities, 1);
  MatrixXd optimal_dual(num_inequalities, 1);
  MatrixXd optimal_y(num_vars, 1);

  optimal_slack.setConstant(1);
  optimal_dual.setConstant(1);
  optimal_slack.topRows(num_inequalities * .5).setZero();
  optimal_dual.bottomRows(num_inequalities - num_inequalities * .5).setZero();
  optimal_y = Eigen::MatrixXd::Random(num_vars, 1);

  C = optimal_slack + A * optimal_y;

  LinearConstraint linear_inequality{A, C};

  MatrixXd eq = MatrixXd::Random(num_equalities, num_vars);
  MatrixXd eq_affine(num_equalities, 1);
  eq = Eigen::MatrixXd::Random(num_equalities, num_vars);
  eq_affine = eq * optimal_y;

  Program prog(num_vars);
  prog.AddConstraint(EqualityConstraints{eq, eq_affine});
  prog.AddConstraint(linear_inequality);

  VectorXd linear_cost(num_vars);
  linear_cost = A.transpose() * optimal_dual;

  VectorXd solution(num_vars);
  Solve(linear_cost, prog, conex::SolverConfiguration(), solution.data());

  EXPECT_NEAR((eq * solution - eq_affine).norm(), 0, 1e-5);
  EXPECT_NEAR((solution - optimal_y).norm(), 0, 1e-5);
}
}  // namespace conex
