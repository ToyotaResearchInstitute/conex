#include <stdlib.h>
#include <iostream>
#include <memory>
#include "conex/cone_program.h"
#include "conex/constraint.h"
#include "conex/equality_constraint.h"
#include "conex/linear_constraint.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

namespace conex {
using DenseMatrix = Eigen::MatrixXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

auto GetConfiguration() {
  SolverConfiguration config;
  config.prepare_dual_variables = true;
  config.inv_sqrt_mu_max = 5e5;
  config.divergence_upper_bound = 1000;
  config.dinf_upper_bound = 1.35;
  config.final_centering_tolerance = 1;
  return config;
}

int DoRandomDenseTest(const SolverConfiguration& config, int number_of_tests,
                      int random_seed) {
  srand(random_seed);
  int total_iters = 0;
  for (int i = 0; i < number_of_tests; i++) {
    int num_variables = 5;
    int num_constraints = 6 + 2 * i;
    double eps = 1e-12;

    DenseMatrix Alinear = DenseMatrix::Random(num_constraints, num_variables);
    DenseMatrix Clinear = DenseMatrix::Random(num_constraints, 1);
    Clinear = Clinear.array().abs();

    LinearConstraint linear_constraint{num_constraints, &Alinear, &Clinear};

    Program prog(num_variables);
    prog.AddConstraint(linear_constraint);

    VectorXd x0 = VectorXd::Random(num_constraints, 1);
    x0 = x0.array().abs();
    x0 *= 0.01 / x0.norm();
    VectorXd b = Alinear.transpose() * x0;
    DenseMatrix y(num_variables, 1);
    Solve(b, prog, config, y.data());

    VectorXd x(num_constraints);
    prog.GetDualVariable(0, &x);

    VectorXd slack = Clinear - Alinear * y;
    EXPECT_LE((Alinear.transpose() * x - b).norm(), eps * b.norm());
    EXPECT_GE(slack.minCoeff(), -eps);
    EXPECT_GE(x.minCoeff(), -eps);
    EXPECT_GE(slack.dot(x), -eps);
    double mu = 1.0 / (config.inv_sqrt_mu_max * config.inv_sqrt_mu_max);
    EXPECT_LE(slack.dot(x), (mu + std::sqrt(eps)) * num_constraints);
    auto status = prog.Status();
    total_iters += status.num_iterations;
  }
  return total_iters;
}

int num_tests = 3;
int random_seed = 1;

GTEST_TEST(KKTSolver, UseIterativeRefinement) {
  auto config = GetConfiguration();
  config.iterative_refinement_iterations = 3;
  config.kkt_solver = CONEX_LLT_FACTORIZATION;
  DoRandomDenseTest(config, num_tests, random_seed);
}

GTEST_TEST(KKTSolver, UseLLT) {
  auto config = GetConfiguration();
  config.iterative_refinement_iterations = 0;
  config.kkt_solver = CONEX_LLT_FACTORIZATION;
  DoRandomDenseTest(config, num_tests, random_seed);
}

GTEST_TEST(LP, UseLDLT) {
  auto config = GetConfiguration();
  config.kkt_solver = CONEX_LDLT_FACTORIZATION;
  DoRandomDenseTest(config, num_tests, random_seed);
}

GTEST_TEST(LP, UseQR) {
  auto config = GetConfiguration();
  config.kkt_solver = CONEX_QR_FACTORIZATION;
  DoRandomDenseTest(config, num_tests, random_seed);
}

GTEST_TEST(QR, SuccessWithDependentInequalityColumns) {
  auto config = GetConfiguration();
  config.kkt_solver = CONEX_QR_FACTORIZATION;
  Eigen::MatrixXd A(3, 4);
  Eigen::VectorXd c(3);
  A << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0;
  c << 1, 1, 1;

  Program prog(4);
  prog.AddConstraint(LinearConstraint{A, c});
  MatrixXd b = A.transpose() * c;

  VectorXd solution(4);
  Solve(b, prog, config, solution.data());
  EXPECT_EQ(prog.Status().solved, 1);

  config.kkt_solver = CONEX_LLT_FACTORIZATION;
  Solve(b, prog, config, solution.data());
  EXPECT_EQ(prog.Status().solved, 0);
}

GTEST_TEST(QR, SuccessWithDependentEquations) {
  auto config = GetConfiguration();
  Eigen::MatrixXd A(3, 4);
  Eigen::VectorXd c(3);
  A << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0;
  c << 1, 1, 1;

  MatrixXd B(5, 4);
  VectorXd d(5);
  B << 1, -1, 0, 0, 1, -1, 0, 0, 2, -2, 0, 0, 3, -3, 0, 0, 1, -1, 0, 0;
  d << 0, 0, 0, 0, 0;

  Program prog(4);
  prog.AddConstraint(LinearConstraint{A, c});
  prog.AddConstraint(EqualityConstraints{B, d});
  MatrixXd b = A.transpose() * c;

  VectorXd solution(4);
  config.kkt_solver = CONEX_QR_FACTORIZATION;
  Solve(b, prog, config, solution.data());
  EXPECT_EQ(prog.Status().solved, 1);
  EXPECT_NEAR((B * solution - d).norm(), 0, 1e-9);

  config.kkt_solver = CONEX_LDLT_FACTORIZATION;
  Solve(b, prog, config, solution.data());
  EXPECT_EQ(prog.Status().solved, 1);
  EXPECT_NEAR((B * solution - d).norm(), 0, 1e-9);
}

}  // namespace conex
