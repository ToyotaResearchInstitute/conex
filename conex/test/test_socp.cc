#include <iostream>
#include <memory>
#include "conex/cone_program.h"
#include "conex/constraint.h"
#include "conex/dense_lmi_constraint.h"
#include "conex/quadratic_cone_constraint.h"
#include "conex/soc_constraint.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

namespace conex {

using DenseMatrix = Eigen::MatrixXd;

int DoMain() {
  int n = 3;
  SolverConfiguration config;
  config.inv_sqrt_mu_max = 10000;

  std::vector<Eigen::MatrixXd> A;
  // 1 x1 x3 x3
  // x1 1
  // x2   1
  // x3     1
  DenseMatrix Wsqrt = Eigen::MatrixXd::Random(n, n);

  Eigen::MatrixXd C(n + 1, n + 1);
  C.setIdentity();
  for (int i = 1; i < n + 1; i++) {
    Eigen::MatrixXd Ai(n + 1, n + 1);
    Ai.setZero();
    Ai.bottomLeftCorner(n, 1) = Wsqrt.col(i - 1);
    Ai.topRightCorner(1, n) = Wsqrt.col(i - 1).transpose();
    A.push_back(Ai);
  }
  DenseLMIConstraint lmi_constraint{n + 1, A, C};

  Eigen::MatrixXd b(n, 1);

  DenseMatrix As(n + 1, n);
  As.setZero();
  As.bottomRightCorner(n, n) = Wsqrt;
  DenseMatrix Cs(n + 1, 1);
  Cs.setConstant(0.000);
  Cs(0) = 1;
  SOCConstraint T(As, Cs);

  QuadraticConstraint soc_constraint(As, Cs);

  DenseMatrix Q = Wsqrt.transpose() * Wsqrt;
  DenseMatrix Aq(n + 1, n);
  Aq.setZero();
  Aq.bottomRightCorner(n, n).setIdentity();
  QuadraticConstraint quad_constraint(Q, Aq, Cs);

  for (int i = -2; i < 2; i++) {
    b.setConstant(i);
    b += Eigen::VectorXd::Random(n, 1) * .02;

    Program prog1(n);
    prog1.AddConstraint(T);
    DenseMatrix y1(n, 1);
    Solve(b, prog1, config, y1.data());

    Program prog2(n);
    prog2.AddConstraint(lmi_constraint);
    DenseMatrix y2(n, 1);
    Solve(b, prog2, config, y2.data());

    EXPECT_NEAR((y1 - y2).norm(), 0, 1e-4);

    Program prog3(n);
    prog3.AddConstraint(quad_constraint);
    DenseMatrix y3(n, 1);
    Solve(b, prog3, config, y3.data());
    EXPECT_NEAR((y1 - y3).norm(), 0, 5e-5);

    Program prog4(n);
    prog4.AddConstraint(soc_constraint);
    DenseMatrix y4(n, 1);
    Solve(b, prog4, config, y4.data());
    EXPECT_NEAR((y1 - y4).norm(), 0, 5e-5);
  }

  return 0;
}

GTEST_TEST(Constraints, SOCP) {
  srand(1);
  for (int i = 0; i < 10; i++) {
    DoMain();
  }
}

}  // namespace conex
