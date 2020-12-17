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
  DenseLMIConstraint T2{n + 1, A, C};

  Eigen::MatrixXd b(n, 1);

  DenseMatrix As(n + 1, n);
  As.setZero();
  As.bottomRightCorner(n, n) = Wsqrt;
  DenseMatrix Cs(n + 1, 1);
  Cs.setZero();
  Cs(0) = 1;
  SOCConstraint T(As, Cs);

  for (int i = -2; i < 2; i++) {
    b.setConstant(i);

    Program prog2;
    prog2.constraints.push_back(T);
    DenseMatrix y2(n, 1);
    Solve(b, prog2, config, y2.data());

    Program prog;
    prog.constraints.push_back(T2);
    DenseMatrix y1(n, 1);
    Solve(b, prog, config, y1.data());

    EXPECT_TRUE((y1 - y2).norm() < 1e-4);

    DenseMatrix Q = Wsqrt.transpose() * Wsqrt;
    DenseMatrix Aq(n + 1, n);
    Aq.setZero();
    Aq.bottomRightCorner(n, n).setIdentity();

    QuadraticConstraint Tq(Q, Aq, Cs);
    Program prog3;
    prog3.constraints.push_back(Tq);
    DenseMatrix y3(n, 1);
    Solve(b, prog3, config, y3.data());
    EXPECT_TRUE((y1 - y3).norm() < 1e-4);
  }

  return 0;
}

TEST(Constraints, SOCP) { DoMain(); }

}  // namespace conex
