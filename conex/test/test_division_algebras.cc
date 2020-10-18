#include "gtest/gtest.h"
#include <Eigen/Dense>
#include "conex/debug_macros.h"
#include "conex/eigen_decomp.h"
#include "conex/jordan_matrix_algebra.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using conex::jordan_algebra::eig;


template<typename T>
void DUMPA(const T& M) {
  for (const auto& e : M) {
    DUMP(e);
  }
}

VectorXd Roots(const VectorXd& x) {
  MatrixXd c(x.rows(), x.rows());
  c.setZero();
  c.bottomRows(1) = -x.transpose();
  c.topRightCorner(x.rows() - 1, x.rows() - 1) = MatrixXd::Identity(x.rows() - 1, x.rows() - 1);
  return eig(c).eigenvalues;
}



template<typename T>
void DoMultTest() {
  using Element = typename T::Element;
  Element x;
  Element y;
  x.setZero();
  x(0) = 1;
  y.setConstant(3);

  Element z;
  T().Multiply(x,y, &z);
  EXPECT_TRUE((y - z).norm() < 1e-8);
}

TEST(JordanMatrixAlgebra, Scalars) {
  DoMultTest<DivisionAlgebra<1>>();
  DoMultTest<DivisionAlgebra<2>>();
  DoMultTest<DivisionAlgebra<4>>();
  DoMultTest<DivisionAlgebra<8>>();
}


template<typename T>
void DoMatrixTest() {
  using Matrix = typename T::Matrix;
  Matrix A = T::Random();
  Matrix B = T::Random();
  auto W =  T().JordanMult(A, B);

  EXPECT_TRUE(T::IsHermitian(B));
  EXPECT_TRUE(T::IsHermitian(A));
  EXPECT_TRUE(T::IsHermitian(W));

  // Test Jordan identity.
  auto Asqr = T().JordanMult(A, A);
  auto BA = T().JordanMult(A, B);
  auto P1 = T().JordanMult(Asqr, BA);

  auto BAsqr = T().JordanMult(B, Asqr);
  auto P2 = T().JordanMult(A, BAsqr);

  EXPECT_TRUE(T().IsEqual(P1, P2));
}


TEST(JordanMatrixAlgebra, Matrices) {
  DoMatrixTest<Octonions>();
  DoMatrixTest<Quaternions>();
  DoMatrixTest<Complex>();
  DoMatrixTest<Real>();
}

template<typename T>
bool DoQuadAssociativeTest() {
  using Matrix = typename T::Matrix;
  Matrix A = T::Random();
  Matrix B = T::Random();
  Matrix Yref = T().QuadRep(B, A);

  Matrix Y1 = T().MatrixMult(T().MatrixMult(B, A), B);
  Matrix Y2 = T().MatrixMult(B, T().MatrixMult(A, B));
  Y1 = T().ScalarMult(Y1, .5);
  Y2 = T().ScalarMult(Y2, .5);
  Y1 = T().MatrixAdd(Y1, Y2);

  return T().IsEqual(Y1, Yref);
}

TEST(JordanMatrixAlgebra, QuadRep) {
  EXPECT_TRUE(DoQuadAssociativeTest<Quaternions>());
  EXPECT_TRUE(DoQuadAssociativeTest<Complex>());
  EXPECT_TRUE(DoQuadAssociativeTest<Real>());
}

Eigen::MatrixXd Sparsity(const Eigen::MatrixXd x) {
  Eigen::MatrixXd y(x.rows(), x.cols());
  for (int i = 0; i < x.rows(); i++) {
    for (int j = 0; j < x.cols(); j++) {
      if (std::fabs(x(i, j)) > 1e-9) {
        y(i, j) = 1;
      } else {
        y(i, j) = 0;
      }
    }
  }
  return y;
}

VectorXd sort(const VectorXd& x) {
  auto y = x;
  std::sort(y.data(), y.data() + x.rows());
  return y;
}

TEST(JordanMatrixAlgebra, EigReal) {
  using T = Real;
  auto Q = T::Random();
  Q = T().JordanMult(Q, Q);

  MatrixXd Xs(3, 3);
  for (int i = 0 ; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Xs(i, j) = Q.at(LinIndex(i, j))(0);
    }
  }

  EXPECT_TRUE((sort(Roots(T().MinimalPolynomial(Q))) -
              sort(eig(Xs).eigenvalues)).norm() < 1e-8);
}

TEST(JordanMatrixAlgebra, Op) {
  using T = Octonions;
  auto Q = T::Random();
  Q = T().JordanMult(Q, Q);

  double normsqr = T().TraceInnerProduct(Q, Q);
  auto eigvals = Roots(T().MinimalPolynomial(Q));

  EXPECT_TRUE(eigvals.minCoeff() > 0);
  EXPECT_TRUE(std::fabs(eigvals.squaredNorm() - normsqr) < 1e-9);
}
