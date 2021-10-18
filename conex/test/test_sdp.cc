#include "conex/cone_program.h"
#include "conex/constraint.h"
#include "conex/dense_lmi_constraint.h"
#include "conex/linear_constraint.h"
#include "conex/test/test_util.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

namespace conex {

using DenseMatrix = Eigen::MatrixXd;

GTEST_TEST(SDP, Mixed) {
  using Eigen::MatrixXd;
  int m = 3;
  int n = 2;
  std::vector<MatrixXd> A(m);
  for (int i = 0; i < m; i++) {
    A.at(i).resize(n, n);
  }
  // clang-format off
  A.at(0) << -1, 0, 
              0, 0;

  A.at(1) << 0, -1, 
             -1, 0;

  A.at(2) << 0, 0,
             0, -1;
  // clang-format on

  Eigen::MatrixXd C = MatrixXd::Zero(n, n);

  Eigen::MatrixXd b(m, 1);
  b << -1, 0, -1;

  Program prog(m);

  Eigen::VectorXd upper_bound(1);
  upper_bound << 1;
  Eigen::VectorXd lower_bound(1);
  lower_bound << 1;
  prog.AddConstraint(UpperBound(upper_bound), {1});
  prog.AddConstraint(LowerBound(lower_bound), {1});
  prog.AddConstraint(DenseLMIConstraint(A, C));

  Eigen::MatrixXd y(m, 1);
  auto config = SolverConfiguration();
  config.max_iterations = 30;
  Solve(b, prog, config, y.data());

  MatrixXd S = MatrixXd::Zero(2, 2);
  for (int i = 0; i < m; i++) {
    S -= y(i) * A.at(i);
  }
  MatrixXd S_expected = MatrixXd::Zero(2, 2);
  S_expected.setConstant(1);
  EXPECT_NEAR((S - S_expected).norm(), 0, 1e-6);
}
int TestDiagonalSDP() {
  srand(1);
  int n = 5;
  int m = 2;
  SolverConfiguration config;
  config.inv_sqrt_mu_max = 25000;
  config.prepare_dual_variables = true;

  DenseMatrix affine2 = DenseMatrix::Identity(n, n);

  DenseMatrix Alinear = DenseMatrix::Random(n, m);
  DenseMatrix Clinear(n, 1);
  Clinear.setConstant(1);

  std::vector<DenseMatrix> constraints2;
  for (int i = 0; i < m; i++) {
    constraints2.push_back(Alinear.col(i).asDiagonal());
  }

  DenseLMIConstraint LMI{n, constraints2, affine2};
  LinearConstraint Linear{n, &Alinear, &Clinear};

  Program prog(m);
  Eigen::VectorXd b(m);
  prog.AddConstraint(LMI, {0, 1});
  b = GetFeasibleObjective(&prog);
  DenseMatrix y1(m, 1);
  Solve(b, prog, config, y1.data());

  Program prog2(m);
  prog2.AddConstraint(Linear, {0, 1});
  b = GetFeasibleObjective(&prog2);
  DenseMatrix y2(m, 1);
  Solve(b, prog2, config, y2.data());

  Program prog3(m);
  DenseMatrix y3(m, 1);
  prog3.AddConstraint(Linear);
  prog3.AddConstraint(Linear);
  Solve(b, prog3, config, y3.data());

  EXPECT_TRUE((y2 - y1).norm() < 1e-6);
  EXPECT_TRUE((y3 - y1).norm() < 1e-4);
  return 0;
}

GTEST_TEST(SDP, DiagonalSDP) {
  for (int i = 0; i < 1; i++) {
    TestDiagonalSDP();
  }
}

GTEST_TEST(SDP, SparseAndDenseAgree) {
  SolverConfiguration config;

  std::vector<int> variables_2{0, 2, 4, 6, 7, 8};
  std::vector<int> variables_1{1, 3, 5};

  int n1 = 5;
  int m1 = variables_1.size();

  int n2 = 5;
  int m2 = variables_2.size();

  int m = m1 + m2;

  vector<DenseMatrix> constraints_1 = GetRandomDenseMatrices(n1, m);
  vector<DenseMatrix> constraints_2 = GetRandomDenseMatrices(n2, m);
  vector<DenseMatrix> sparse_constraints_1;
  vector<DenseMatrix> sparse_constraints_2;

  for (const auto i : variables_1) {
    sparse_constraints_1.push_back(constraints_1.at(i));
    constraints_2.at(i).setZero();
  }

  for (const auto i : variables_2) {
    sparse_constraints_2.push_back(constraints_2.at(i));
    constraints_1.at(i).setZero();
  }

  DenseMatrix affine_1 = Eigen::MatrixXd::Identity(n1, n1);
  DenseMatrix affine_2 = Eigen::MatrixXd::Identity(n2, n2);
  DenseLMIConstraint LMI1{n1, constraints_1, affine_1};
  DenseLMIConstraint LMI2{n2, constraints_2, affine_2};

  Program prog(m);
  prog.AddConstraint(LMI1);
  prog.AddConstraint(LMI2);

  auto b = GetFeasibleObjective(&prog);

  DenseMatrix y(m1 + m2, 1);
  int success = Solve(b, prog, config, y.data());
  EXPECT_EQ(success, 1);

  DenseLMIConstraint sparse_LMI1{sparse_constraints_1, affine_1};
  DenseLMIConstraint sparse_LMI2{sparse_constraints_2, affine_2};

  Program sparse_prog(m);
  sparse_prog.AddConstraint(sparse_LMI1, variables_1);
  sparse_prog.AddConstraint(sparse_LMI2, variables_2);

  DenseMatrix y_sparse(m1 + m2, 1);
  success = Solve(b, sparse_prog, config, y_sparse.data());
  EXPECT_EQ(success, 1);

  EXPECT_NEAR((y - y_sparse).norm(), 0, 1e-8);
}

int TestSDP(int n, int m) {
  SolverConfiguration config;
  auto constraints2 = GetRandomDenseMatrices(n, m);

  DenseMatrix affine2 = Eigen::MatrixXd::Identity(n, n);
  DenseLMIConstraint LMI{n, constraints2, affine2};

  Program prog(m);
  DenseMatrix y(m, 1);
  prog.AddConstraint(LMI);

  auto b = GetFeasibleObjective(&prog);
  config.prepare_dual_variables = 1;
  Solve(b, prog, config, y.data());

  DenseMatrix x(n, n);
  prog.GetDualVariable(0, &x);

  DenseMatrix slack = affine2;
  DenseMatrix res = b;
  for (int i = 0; i < m; i++) {
    slack -= constraints2.at(i) * y(i);
    res(i) -= (constraints2.at(i) * x).trace();
  }

  EXPECT_NEAR(eig(slack).eigenvalues.minCoeff(), 0, 1e-5);
  EXPECT_NEAR(res.norm(), 0, 1e-8);
  EXPECT_NEAR((slack * x).trace(), 0, 1e-4);

  return 0;
}

GTEST_TEST(SDP, ProfileSDP) {
  for (int i = 1; i < 10; i++) {
    for (int j = i; j < i + 10; j++) {
      TestSDP(j /*order n*/, i /*num vars*/);
    }
  }
}
}  // namespace conex
