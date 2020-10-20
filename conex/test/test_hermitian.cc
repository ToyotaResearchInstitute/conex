#include "conex/hermitian_psd.h"
#include "conex/dense_lmi_constraint.h"

#include "gtest/gtest.h"
#include <Eigen/Dense>

#include "conex/cone_program.h"

using Eigen::MatrixXd;

MatrixXd ToMat(const Real::Matrix& x) {
  MatrixXd y(3, 3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      y(i, j) = x.at(LinIndex(i, j))(0);
    }
  }
  return y;
}

std::vector<MatrixXd> ToMat(const std::vector<Real::Matrix>& x) {
  std::vector<MatrixXd> y;
  for (const auto& e : x) {
    y.push_back(ToMat(e));
  }
  return y;
}


int TestSDP() {
  using T = Real;
  using Matrix = typename T::Matrix;
  SolverConfiguration config;
  int n = 3;
  int m = 2;
  std::vector<Matrix> constraint_matrices(m);
  Matrix constraint_affine = T::Identity();
  for (int i = 0; i < m; i++) {
    constraint_matrices.at(i) = T::Random(); 
  }
  HermitianPsdConstraint<T> T2(3, constraint_matrices, constraint_affine);

  Program prog;
  DenseMatrix y(m, 1);
  prog.constraints.push_back(T2);


  auto b = GetFeasibleObjective(m, prog.constraints);
  Solve(b, prog, config, y.data());

  Program prog2;
  DenseMatrix y2(m, 1);
  prog2.constraints.push_back(DenseLMIConstraint(3, ToMat(constraint_matrices),
                                                 ToMat(constraint_affine)));

  Solve(b, prog2, config, y2.data());

  return 1;

  DenseMatrix x(n, n);
  prog.constraints.at(0).get_dual_variable(x.data());
  x.array() /= prog.stats.sqrt_inv_mu[prog.stats.num_iter - 1];

  Matrix slack = constraint_affine;
  DenseMatrix res = b;
  for (int i = 0; i < m; i++) {
    slack = T().MatrixAdd(slack, T::ScalarMult(constraint_matrices.at(i), -y(i)));
 //   b(i) -= (constraint_matrices.at(i) * x).trace();
  }

  EXPECT_TRUE(eigenvalues<T>(slack).minCoeff() > 1e-8);
  //EXPECT_TRUE(b.norm() < 1e-8);
  //EXPECT_TRUE((slack*x).trace() < 1e-4);

  return 0;
}

TEST(SDP, RandomSDP) {
  for (int i = 0; i < 1; i++) {
    TestSDP(); 
  }
}
