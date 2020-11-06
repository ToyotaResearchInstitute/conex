#include "conex/vect_jordan_matrix_algebra.h"

#include "gtest/gtest.h"
#include <Eigen/Dense>

#include "conex/debug_macros.h"
#include "conex/eigen_decomp.h"

namespace conex {

using Eigen::VectorXd;
using Eigen::MatrixXd;
using conex::jordan_algebra::eig;

VectorXd sort(const VectorXd& x) {
  auto y = x;
  std::sort(y.data(), y.data() + x.rows());
  return y;
}

template<typename T>
bool DoMultiplyByIdentity() {
  auto X = T::Random(3, 3);
  auto I = T::Identity(3);
  return T::IsEqual(X, T::Multiply(X, I)); 
}

TEST(JordanMatrixAlgebra, MultiplyByIdentity) {
  EXPECT_TRUE(DoMultiplyByIdentity<Real>());
  EXPECT_TRUE(DoMultiplyByIdentity<Complex>());
  EXPECT_TRUE(DoMultiplyByIdentity<Quaternions>());
  EXPECT_TRUE(DoMultiplyByIdentity<Octonions>());
}

template<typename T>
bool DoVerifyJordanIdentity(int n) {
  using Matrix = typename T::Matrix;
  Matrix A = T::Random(n, n); A = T::Add(A, T::ConjugateTranspose(A));
  Matrix B = T::Random(n, n); B = T::Add(B, T::ConjugateTranspose(B));
  auto W =  T::JordanMultiply(A, B);

  EXPECT_TRUE(T::IsHermitian(B));
  EXPECT_TRUE(T::IsHermitian(A));
  EXPECT_TRUE(T::IsHermitian(W));

  // Test Jordan identity.
  auto Asqr = T::JordanMultiply(A, A);
  auto BA = T::JordanMultiply(A, B);
  auto P1 = T::JordanMultiply(Asqr, BA);

  auto BAsqr = T::JordanMultiply(B, Asqr);
  auto P2 = T::JordanMultiply(A, BAsqr);

  return T::IsEqual(P1, P2);
}

TEST(JordanMatrixAlgebra, VerifyJordanIdentity) {
  EXPECT_TRUE(DoVerifyJordanIdentity<Octonions>(3));
  EXPECT_TRUE(DoVerifyJordanIdentity<Octonions>(2));
  EXPECT_TRUE(!DoVerifyJordanIdentity<Octonions>(4));

  EXPECT_TRUE(DoVerifyJordanIdentity<Quaternions>(5));
  EXPECT_TRUE(DoVerifyJordanIdentity<Complex>(5));
  EXPECT_TRUE(DoVerifyJordanIdentity<Real>(5));
}

template<typename T>
bool DoQuadraticRepresentationAssociativeTest(int d) {
  using Matrix = typename T::Matrix;
  Matrix A = T::Random(d, d); A = T::Add(A, T::ConjugateTranspose(A));
  Matrix B = T::Random(d, d); B = T::Add(B, T::ConjugateTranspose(B));
  Matrix Yref = T::QuadraticRepresentation(B, A);

  Matrix Y1 = T::Multiply(T::Multiply(B, A), B);
  Matrix Y2 = T::Multiply(B, T::Multiply(A, B));
  Y1 = T::ScalarMultiply(Y1, .5);
  Y2 = T::ScalarMultiply(Y2, .5);
  Y1 = T::Add(Y1, Y2);

  return T::IsEqual(Y1, Yref);
}

TEST(JordanMatrixAlgebra, QuadraticRepresentationAssociativeTest) {
  EXPECT_TRUE(!DoQuadraticRepresentationAssociativeTest<Octonions>(3));
  EXPECT_TRUE(DoQuadraticRepresentationAssociativeTest<Quaternions>(5));
  EXPECT_TRUE(DoQuadraticRepresentationAssociativeTest<Complex>(5));
  EXPECT_TRUE(DoQuadraticRepresentationAssociativeTest<Real>(5));
}

template<typename T>
void DoEigenvalueTests() {
  double eps = 1e-9;
  using Matrix = typename T::Matrix;
  int n = 3;
  Matrix Q = T::Random(n, n); Q = T::Add(Q, T::ConjugateTranspose(Q));
  Q = T::JordanMultiply(Q, Q);

  auto I = T::Identity(n);
  double normsqr = T::TraceInnerProduct(Q, Q);
  double trace = T::TraceInnerProduct(I, Q);
  auto eigvals = T::Eigenvalues(Q);

  EXPECT_TRUE(eigvals.minCoeff() > eps);
  EXPECT_NEAR(eigvals.squaredNorm(), normsqr, eps);
  EXPECT_NEAR(eigvals.sum(), trace, eps);

  // eigvals = T::Eigenvalues(I);
  // for (int i = 0; i < eigvals.rows(); i++) {
  //   EXPECT_NEAR(eigvals(i), 1, 1e-8);
  // }
}

TEST(JordanMatrixAlgebra, EigenvaluePropertiesReal) {
  DoEigenvalueTests<Real>();
}

TEST(JordanMatrixAlgebra, EigenvaluePropertiesComplex) {
  DoEigenvalueTests<Complex>();
}

TEST(JordanMatrixAlgebra, EigenvaluePropertiesQuaternion) {
  DoEigenvalueTests<Quaternions>();
}

TEST(JordanMatrixAlgebra, EigenvaluePropertiesOctonion) {
  DoEigenvalueTests<Octonions>();
}

TEST(JordanMatrixAlgebra, HermitianRealMatchesEigen) {
  using T = Real;
  int n = 3;
  auto Q = T::Random(n, n);
  Q = T::JordanMultiply(Q, Q);

  EXPECT_TRUE((T::Eigenvalues(Q) - sort(eig(Q.at(0)).eigenvalues)).norm() < 1e-8);
}

template<int n>
bool DoTestOrthogonal(int d) {
  double eps = 1e-8;
  using T = MatrixAlgebra<n>;
  auto Q = T::Random(d, d);

  Q = T::Orthogonalize(Q);

  auto I = T::ScalarMultiply(T::Identity(d), -1);
  auto res = T::Add(I,   T::Multiply(Q, T::ConjugateTranspose(Q)));
  return T::TraceInnerProduct(res, res) < eps;
}

TEST(JordanMatrixAlgebra, Orthogonal) {
  EXPECT_TRUE(DoTestOrthogonal<1>(4));
  EXPECT_TRUE(DoTestOrthogonal<2>(4));
  EXPECT_TRUE(DoTestOrthogonal<4>(4));
}


template<int n>
void DoEigenvaluesFromSpectralDecomp(int d) {
  double eps = 1e-8;
  using T = MatrixAlgebra<n>;
  auto Q = T::Random(d, d);

  Q = T::Orthogonalize(Q);
  HyperComplexMatrix D = T::Zero(d, d);
  for (int i = 0; i < d; i++) {
    D.at(0)(i, i) = -d/2 + i;
  }
  auto X = T::Multiply(T::Multiply(Q, D), T::ConjugateTranspose(Q));

  VectorXd calc = sort(T::Eigenvalues(X));
  for (int i = 0; i < d; i++) {
    EXPECT_NEAR(calc(i), D.at(0)(i, i), eps);
  }

  auto r = T::Random(d, 1);
  calc = sort(T::ApproximateEigenvalues(X, r, d));
  for (int i = 0; i < d; i++) {
    EXPECT_NEAR(calc(i), D.at(0)(i, i), eps);
  }
}

TEST(JordanMatrixAlgebra, EigenvaluesFromSpectralDecomp) {
  DoEigenvaluesFromSpectralDecomp<1>(4);
  DoEigenvaluesFromSpectralDecomp<2>(4);
  DoEigenvaluesFromSpectralDecomp<4>(4);
}


template<int n>
void DoAsymmetricEigenvaluesTest(int d) {
  double eps = 1e-8;
  using T = MatrixAlgebra<n>;
  auto Wsqrt = T::Random(d, d); Wsqrt = T::Add(Wsqrt, T::ConjugateTranspose(Wsqrt));
  auto S = T::Random(d, d); S = T::Add(S, T::ConjugateTranspose(S));

  auto W = T::Multiply(Wsqrt, T::ConjugateTranspose(Wsqrt)); 


  VectorXd ref = sort(T::Eigenvalues(T::QuadraticRepresentation(Wsqrt, S)));
  VectorXd calc = sort(T::ApproximateEigenvalues(T::Multiply(W, S), W, T::Random(d, 1), d ));
  for (int i = 0; i < d; i++) {
    EXPECT_NEAR(calc(i), ref(i), eps);
  }

}

TEST(JordanMatrixAlgebra, AsymmetricEigenvaluesTest) {
  DoAsymmetricEigenvaluesTest<1>(4);
  DoAsymmetricEigenvaluesTest<2>(4);
  DoAsymmetricEigenvaluesTest<4>(4);
}



} // namespace conex

