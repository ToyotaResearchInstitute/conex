#define EIGEN_RUNTIME_NO_MALLOC
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include "conex/linear_constraint.h"
#include "conex/constraint.h"
#include "conex/cone_program.h"

#include "conex/conex.h"

using DenseMatrix = Eigen::MatrixXd;
using Eigen::VectorXd;
TEST(message_test,content) {
  for (int i = 0; i < 1; i++) {
    ConexSolverConfiguration config = ConexDefaultOptions();
    config.prepare_dual_variables = true;
    config.inv_sqrt_mu_max = 1000000;
    int m = 5;
    int n = 6 + 2 * i;
    double eps = 1e-12;

    DenseMatrix Alinear = DenseMatrix::Random(n, m);
    DenseMatrix Clinear(n, 1);
    Clinear.setConstant(1);

    DenseMatrix affine2 = Clinear.asDiagonal();
    LinearConstraint T3{n, &Alinear, &Clinear};

    Program prog;
    prog.constraints.push_back(T3);

    auto b = GetFeasibleObjective(m, prog.constraints);
    DenseMatrix y(m, 1);
    // Eigen::internal::set_is_malloc_allowed(false);
    Solve(b, prog,  config, y.data());
    // Eigen::internal::set_is_malloc_allowed(true);

    VectorXd x(n);
    prog.constraints.at(0).get_dual_variable(x.data());
    x.array() /= prog.stats.sqrt_inv_mu[prog.stats.num_iter - 1];

    VectorXd slack = Clinear-Alinear*y;
    EXPECT_TRUE((Alinear.transpose()*x - b).norm() <= eps);
    EXPECT_TRUE((slack).minCoeff() >= -eps);


    double sqrtmu = 1.0/prog.stats.sqrt_inv_mu[prog.stats.num_iter - 1];
    EXPECT_TRUE(slack.dot(x) <= sqrtmu*sqrtmu * n + eps) ;
  }
}
