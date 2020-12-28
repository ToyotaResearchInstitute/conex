#include <stdlib.h>
#include <iostream>
#include <memory>
#include "conex/cone_program.h"
#include "conex/constraint.h"
#include "conex/linear_constraint.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

namespace conex {
using DenseMatrix = Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(LP, Dense) {
  for (int i = 0; i < 1; i++) {
    SolverConfiguration config;
    config.prepare_dual_variables = true;
    config.inv_sqrt_mu_max = 1000000;

    int m = 5;
    int n = 6 + 2 * i;
    double eps = 1e-12;

    DenseMatrix Alinear = DenseMatrix::Random(n, m);
    DenseMatrix Clinear(n, 1);
    Clinear.setConstant(1);

    LinearConstraint linear_constraint{n, &Alinear, &Clinear};

    Program prog(m);
    prog.SetNumberOfVariables(m);
    prog.AddConstraint(linear_constraint);

    VectorXd b = Alinear.transpose() * Clinear;
    DenseMatrix y(m, 1);
    Solve(b, prog, config, y.data());

    VectorXd x(n);
    prog.GetDualVariable(0, &x);

    VectorXd slack = Clinear - Alinear * y;
    EXPECT_TRUE((Alinear.transpose() * x - b).norm() <= eps);
    EXPECT_TRUE((slack).minCoeff() >= -eps);
  }
}

Eigen::VectorXd Vars(const Eigen::VectorXd& x, std::vector<int> indices) {
  Eigen::VectorXd z(indices.size());
  int cnt = 0;
  for (auto i : indices) {
    z(cnt++) = x(i);
  }
  return z;
}

using Eigen::MatrixXd;
using std::vector;
auto Combine(vector<MatrixXd> A, vector<MatrixXd> C,
             vector<vector<int> > vars) {
  int n = A.at(0).rows() * A.size();
  int m = A.at(0).cols() * A.size();
  Eigen::MatrixXd Af(n, m);
  Eigen::MatrixXd Cf(n, 1);
  Af.setZero();
  Cf.setZero();
  int cnt = 0;
  int max_v = 0;
  for (size_t i = 0; i < A.size(); i++) {
    for (int k = 0; k < A.at(i).rows(); k++) {
      Cf(cnt, 0) = C.at(i)(k, 0);
      for (size_t j = 0; j < vars.at(i).size(); j++) {
        Af(cnt, vars.at(i).at(j)) = A.at(i)(k, j);
        if (vars.at(i).at(j) > max_v) {
          max_v = vars.at(i).at(j);
        }
      }
      cnt++;
    }
  }

  Eigen::MatrixXd Ac = Af.topLeftCorner(cnt, max_v + 1);
  Eigen::MatrixXd Cc = Cf.topLeftCorner(cnt, 1);
  return LinearConstraint{Ac, Cc};
}

Eigen::VectorXd SolveSparseHelper(bool sparse) {
  double eps = 1e-8;
  using Eigen::MatrixXd;
  using std::vector;
  SolverConfiguration config;
  config.prepare_dual_variables = true;

  int number_of_constraints = 50;
  std::vector<std::vector<int> > variables(number_of_constraints);
  vector<MatrixXd> A(number_of_constraints);
  vector<MatrixXd> C(number_of_constraints);

  srand(1);
  int number_of_variables;
  int vars_per_constraint = 5;
  int rows_per_constraint = 10;
  int var_start = 0;
  for (int i = 0; i < number_of_constraints; i++) {
    for (int j = 0; j < vars_per_constraint; j++) {
      variables.at(i).push_back(var_start + j);
    }
    var_start = variables.at(i).back();
  }
  number_of_variables = variables.back().back() + 1;
  for (int i = 0; i < number_of_constraints; i++) {
    MatrixXd Ai =
        DenseMatrix::Random(rows_per_constraint, variables.at(i).size());
    MatrixXd Ci(rows_per_constraint, 1);
    Ci.setConstant(1 * (i * .01 + 1));
    C.at(i) = Ci;
    A.at(i) = Ai;
  }

  auto E = C.at(0);
  E.setConstant(1);
  Eigen::VectorXd b(number_of_variables);
  b.setZero();
  for (int i = 0; i < number_of_constraints; i++) {
    Eigen::VectorXd bi = A.at(i).transpose() * E;
    int cnt = 0;
    for (auto k : variables.at(i)) {
      b(k) += bi(cnt++);
    }
  }

  Program prog(number_of_variables);

  MatrixXd y(number_of_variables, 1);
  if (sparse) {
    for (int i = 0; i < number_of_constraints; i++) {
      prog.AddConstraint(LinearConstraint(A.at(i), C.at(i)), variables.at(i));
    }

    Solve(b, prog, config, y.data());
    MatrixXd Ax = b * 0;
    for (int i = 0; i < number_of_constraints; i++) {
      VectorXd slack = C.at(i) - A.at(i) * Vars(y, variables.at(i));
      EXPECT_TRUE((slack).minCoeff() >= -eps);

      MatrixXd xi(A.at(i).rows(), 1);
      prog.GetDualVariable(i, &xi);

      VectorXd temp = A.at(i).transpose() * xi;
      int k = 0;
      for (auto vk : variables.at(i)) {
        Ax(vk) += temp(k);
        k++;
      }
    }
    EXPECT_NEAR((Ax - b).norm(), 0, 1e-8);
  } else {
    prog.AddConstraint(Combine(A, C, variables));
    Solve(b, prog, config, y.data());

    auto res = b;
    auto L = Combine(A, C, variables);
    VectorXd slack = L.constraint_affine_ - L.constraint_matrix_ * y;

    MatrixXd xi(L.constraint_affine_.rows(), 1);
    prog.GetDualVariable(0, &xi);
    EXPECT_NEAR((b - L.constraint_matrix_.transpose() * xi).norm(), 0, 1e-8);
  }

  return y;
}

// Solve the same problem in dense and sparse format.
TEST(LP, Sparse) {
  auto y1 = SolveSparseHelper(true);
  auto y2 = SolveSparseHelper(false);
  EXPECT_NEAR((y1 - y2).norm(), 0, 1e-7);
}

Eigen::VectorXd SolveFillIn(bool sparse) {
  double eps = 1e-8;
  using Eigen::MatrixXd;
  using std::vector;
  SolverConfiguration config;
  config.prepare_dual_variables = true;

  int number_of_constraints = 4;
  std::vector<std::vector<int> > variables{{0, 1}, {1, 2}, {2, 3}, {0, 3}};

  int number_of_variables = 3 + 1;
  vector<MatrixXd> A(number_of_constraints);
  vector<MatrixXd> C(number_of_constraints);

  srand(1);
  int rows_per_constraint = 3;

  for (int i = 0; i < number_of_constraints; i++) {
    MatrixXd Ai =
        DenseMatrix::Random(rows_per_constraint, variables.at(i).size());
    MatrixXd Ci(rows_per_constraint, 1);
    Ci.setConstant(1 * (i * .01 + 1));
    C.at(i) = Ci;
    A.at(i) = Ai;
  }

  auto E = C.at(0);
  E.setConstant(1);
  Eigen::VectorXd b(number_of_variables);
  b.setZero();
  for (int i = 0; i < number_of_constraints; i++) {
    Eigen::VectorXd bi = A.at(i).transpose() * E;
    int cnt = 0;
    for (auto k : variables.at(i)) {
      b(k) += bi(cnt++);
    }
  }

  Program prog(number_of_variables);

  MatrixXd y(number_of_variables, 1);
  if (sparse) {
    for (int i = 0; i < number_of_constraints; i++) {
      prog.AddConstraint(LinearConstraint(A.at(i), C.at(i)), variables.at(i));
    }

    Solve(b, prog, config, y.data());
    MatrixXd Ax = b * 0;
    for (int i = 0; i < number_of_constraints; i++) {
      VectorXd slack = C.at(i) - A.at(i) * Vars(y, variables.at(i));
      EXPECT_TRUE((slack).minCoeff() >= -eps);

      MatrixXd xi(A.at(i).rows(), 1);
      prog.GetDualVariable(i, &xi);
      VectorXd temp = A.at(i).transpose() * xi;
      int k = 0;
      for (auto vk : variables.at(i)) {
        Ax(vk) += temp(k);
        k++;
      }
    }
    EXPECT_NEAR((Ax - b).norm(), 0, 1e-8);
  } else {
    prog.AddConstraint(Combine(A, C, variables));
    Solve(b, prog, config, y.data());

    auto res = b;
    auto L = Combine(A, C, variables);
    VectorXd slack = L.constraint_affine_ - L.constraint_matrix_ * y;

    MatrixXd xi(L.constraint_affine_.rows(), 1);
    prog.GetDualVariable(0, &xi);
    EXPECT_NEAR((b - L.constraint_matrix_.transpose() * xi).norm(), 0, 1e-8);
  }

  return y;
}

TEST(LP, SparseWithFillIn) {
  DUMP("HEHEH!");
  auto y1 = SolveFillIn(true);
  auto y2 = SolveFillIn(false);
  EXPECT_NEAR((y1 - y2).norm(), 0, 1e-7);
}

}  // namespace conex
