#include "conex/quadratic_cost.h"

#include "gtest/gtest.h"

#include "conex/cone_program.h"
#include "conex/debug_macros.h"
#include "conex/linear_constraint.h"
#include "conex/quadratic_cone_constraint.h"
#include "conex/soc_constraint.h"

namespace conex {

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct ProblemData {
  Eigen::MatrixXd W;
  Eigen::MatrixXd Wsqrt;
  Eigen::VectorXd c;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  Eigen::VectorXd optimal_slack;
  Eigen::VectorXd optimal_lambda;
  Eigen::VectorXd optimal_x;
};

MatrixXd NormalizeRows(const MatrixXd& X) {
  MatrixXd Y = X;
  for (int i = 0; i < X.rows(); i++) {
    Y.row(i) = X.row(i) / X.row(i).norm();
  }
  Y = Y / sqrt((double)X.rows());
  return Y;
}

ProblemData RandomWellPosedProblem(int n, int num_ineqs,
                                   int rank_of_quadratic) {
  if (rank_of_quadratic + num_ineqs < n) {
    throw std::runtime_error(
        "Must have rank of quadratic + num_ineqs >= num_vars");
  }
  if (rank_of_quadratic > n) {
    throw std::runtime_error("Must have rank of quadratic <= num_vars");
  }

  ProblemData data;

  data.A = NormalizeRows(MatrixXd::Random(num_ineqs, n));
  if (rank_of_quadratic > 0) {
    data.Wsqrt = NormalizeRows(MatrixXd::Random(rank_of_quadratic, n));
    data.W = data.Wsqrt.transpose() * data.Wsqrt;
  } else {
    data.Wsqrt = MatrixXd::Random(1, n) * 0;
    data.W = data.Wsqrt.transpose() * data.Wsqrt;
  }

  VectorXd strictly_feasible_slack(num_ineqs);
  strictly_feasible_slack.setConstant(1);
  VectorXd strictly_feasible_lambda(num_ineqs);
  strictly_feasible_lambda.setConstant(1);
  strictly_feasible_slack += VectorXd::Random(num_ineqs) * .1 / sqrt(num_ineqs);
  strictly_feasible_lambda +=
      VectorXd::Random(num_ineqs) * .1 / sqrt(num_ineqs);

  VectorXd feasible_x = VectorXd::Random(n);
  feasible_x = feasible_x / feasible_x.norm();

  data.b = strictly_feasible_slack - (data.A * feasible_x);
  data.b.array();
  data.c = data.A.transpose() * strictly_feasible_lambda - data.W * feasible_x;

  return data;
}

// Given x and complementarity lambda and slack,
// we construct QP
//
//   min.       x'Qx + c'x
//   subject to Ax + b >= 0
//
// To do this, we randomly sample A and W and set (c, b)
// via:
//
//   c = A' * lambda - Wx
//   b = s - A * x
//
ProblemData ProblemDataWithSolution(int n, int num_ineqs) {
  ProblemData data;
  int size_of_active_set = n;

  VectorXd optimal_slack(num_ineqs);
  optimal_slack.setZero();
  VectorXd optimal_lambda(num_ineqs);
  optimal_lambda.setZero();
  optimal_lambda.head(size_of_active_set) =
      VectorXd::Random(size_of_active_set).array().abs();
  optimal_slack.tail(num_ineqs - size_of_active_set) =
      VectorXd::Random(num_ineqs - size_of_active_set).array().abs();

  optimal_lambda.head(size_of_active_set)
      .setLinSpaced(size_of_active_set, 1, size_of_active_set);
  optimal_slack.tail(num_ineqs - size_of_active_set).setConstant(1);

  VectorXd optimal_x = VectorXd::Random(n);
  data.W = MatrixXd::Identity(n, n);
  data.A = MatrixXd::Random(num_ineqs, n);
  data.b = optimal_slack - data.A * optimal_x;
  data.c = data.A.transpose() * optimal_lambda - data.W * optimal_x;

  data.optimal_slack = optimal_slack;
  data.optimal_lambda = optimal_lambda;
  data.optimal_x = optimal_x;

  return data;
}

void SolveQPInstance(ProblemData& data, const SolverConfiguration& config,
                     bool print_stats = false) {
  int num_vars = data.A.cols();
  Program prog(num_vars);
  VectorXd solution(num_vars);

  std::vector<int> vars;
  for (int i = 0; i < num_vars; i++) {
    vars.push_back(i);
  }

  prog.AddQuadraticCost(data.W, vars);
  prog.AddLinearCost(data.c);
  // Ax <= b.
  prog.AddConstraint(LinearConstraint(-data.A, data.b), vars);

  bool error = !Solve(prog, config, solution.data());
  EXPECT_NEAR((solution - data.optimal_x).norm(), 0.0, 1e-9);
  EXPECT_NEAR((data.A * solution + data.b - data.optimal_slack).norm(), 0.0,
              1e-9);
  EXPECT_EQ(error, false);
}

}  // namespace conex

void SolveRandomQP(int num_vars, int num_ineqs) {
  conex::SolverConfiguration config;

  config.enable_line_search = true;
  config.initial_centering_steps_coldstart = 0;
  config.enable_rescaling = false;
  config.inv_sqrt_mu_max = 1e7;
  config.max_iterations = 25;
  config.final_centering_tolerance = 1.05;
  config.final_centering_steps = 0;
  config.minimum_mu = 0;
  config.kkt_error_tolerance = 1e45;
  config.dinf_upper_bound = 1;
  config.prepare_dual_variables = 1;

  conex::ProblemData data = conex::ProblemDataWithSolution(num_vars, num_ineqs);

  conex::SolveQPInstance(data, config);
}

GTEST_TEST(RandomQP, Small) {
  int num_vars = 5;
  int num_ineqs = 10;
  SolveRandomQP(num_vars, num_ineqs);
}

GTEST_TEST(RandomQP, Medium) {
  int num_vars = 10;
  int num_ineqs = 20;
  SolveRandomQP(num_vars, num_ineqs);
}

GTEST_TEST(RandomQP, Large) {
  int num_vars = 50;
  int num_ineqs = 70;
  SolveRandomQP(num_vars, num_ineqs);
}
