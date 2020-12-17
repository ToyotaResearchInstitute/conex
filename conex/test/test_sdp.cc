#include "conex/cone_program.h"
#include "conex/constraint.h"
#include "conex/dense_lmi_constraint.h"
#include "conex/eigen_decomp.h"
#include "conex/linear_constraint.h"
#include "conex/test/test_util.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

namespace conex {

using DenseMatrix = Eigen::MatrixXd;
#define TEST_OR_PROFILE 1
#if TEST_OR_PROFILE
int TestDiagonalSDP() {
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

  Program prog;
  prog.constraints.push_back(LMI);
  auto b = GetFeasibleObjective(m, prog.constraints);
  DenseMatrix y1(m, 1);
  Solve(b, prog, config, y1.data());

  Program prog2;
  prog2.constraints.push_back(Linear);
  DenseMatrix y2(m, 1);
  Solve(b, prog2, config, y2.data());

  Program prog3;
  DenseMatrix y3(m, 1);
  prog3.constraints.push_back(Linear);
  prog3.constraints.push_back(Linear);
  Solve(b, prog3, config, y3.data());

  EXPECT_TRUE((y2 - y1).norm() < 1e-6);
  EXPECT_TRUE((y3 - y1).norm() < 1e-4);
  return 0;
}

TEST(SDP, DiagonalSDP) {
  for (int i = 0; i < 1; i++) {
    TestDiagonalSDP();
  }
}

TEST(SDP, SparseAndDenseAgree) {
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

  Program prog;
  prog.constraints.push_back(LMI1);
  prog.constraints.push_back(LMI2);

  auto b = GetFeasibleObjective(m, prog.constraints);

  DenseMatrix y(m1 + m2, 1);
  int success = Solve(b, prog, config, y.data());
  EXPECT_EQ(success, 1);

  SparseLMIConstraint sparse_LMI1{sparse_constraints_1, affine_1, variables_1};
  SparseLMIConstraint sparse_LMI2{sparse_constraints_2, affine_2, variables_2};

  Program sparse_prog;
  sparse_prog.constraints.push_back(sparse_LMI1);
  sparse_prog.constraints.push_back(sparse_LMI2);

  DenseMatrix y_sparse(m1 + m2, 1);
  success = Solve(b, sparse_prog, config, y_sparse.data());
  EXPECT_EQ(success, 1);

  EXPECT_NEAR((y - y_sparse).norm(), 0, 1e-8);
}
#else

int TestSDP(int i) {
  SolverConfiguration config;
  int n = 150;
  int m = 50;
  auto constraints2 = GetRandomDenseMatrices(n, m);

  DenseMatrix affine2 = Eigen::MatrixXd::Identity(n, n);
  DenseLMIConstraint LMI{n, constraints2, affine2};

  Program prog;
  DenseMatrix y(m, 1);
  prog.constraints.push_back(LMI);

  auto b = GetFeasibleObjective(m, prog.constraints);
  Solve(b, prog, config, y.data());
  return 0;

  DenseMatrix x(n, n);
  prog.constraints.at(0).get_dual_variable(x.data());
  x.array() /= prog.stats.sqrt_inv_mu[prog.stats.num_iter - 1];

  DenseMatrix slack = affine2;
  DenseMatrix res = b;
  for (int i = 0; i < m; i++) {
    slack -= constraints2.at(i) * y(i);
    res(i) -= (constraints2.at(i) * x).trace();
  }

  EXPECT_TRUE(conex::jordan_algebra::eig(slack).eigenvalues.minCoeff() > 1e-8);
  EXPECT_TRUE(res.norm() < 1e-7);
  EXPECT_TRUE((slack * x).trace() < 1e-4);

  return 0;
}

TEST(SDP, ProfileSDP) {
  for (int i = 0; i < 1; i++) {
    TestSDP(i);
  }
}
#endif

}  // namespace conex
