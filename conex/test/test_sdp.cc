#include "gtest/gtest.h"
#include <Eigen/Dense>
#include "conex/linear_constraint.h"
#include "conex/dense_lmi_constraint.h"
#include "conex/constraint.h"
#include "conex/cone_program.h"
#include "conex/test/test_util.h"
#include "conex/eigen_decomp.h"

using DenseMatrix = Eigen::MatrixXd;

int TestDiagonalSDP() {
  int n = 20;
  int m = 10;
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

int TestSDP() {
  SolverConfiguration config;
  int n = 15;
  int m = 13;
  auto constraints2 = GetRandomDenseMatrices(n, m);

  DenseMatrix affine2 = Eigen::MatrixXd::Identity(n, n);
  DenseLMIConstraint LMI{n, constraints2, affine2};

  Program prog;
  DenseMatrix y(m, 1);
  prog.constraints.push_back(LMI);


  auto b = GetFeasibleObjective(m, prog.constraints);
  Solve(b, prog, config, y.data());

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
  EXPECT_TRUE((slack*x).trace() < 1e-4);

  return 0;
}



TEST(SDP, DiagonalSDP) {
  for (int i = 0; i < 1; i++) {
    TestDiagonalSDP();
  }
}

TEST(SDP, RandomSDP) {
  for (int i = 0; i < 1; i++) {
    TestSDP(); }
}
