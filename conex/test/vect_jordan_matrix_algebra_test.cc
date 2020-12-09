#include "conex/vect_jordan_matrix_algebra.h"

#include "gtest/gtest.h"
#include <Eigen/Dense>

#include "conex/debug_macros.h"
#include "conex/eigen_decomp.h"

namespace conex {





using conex::jordan_algebra::eig;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using JordanTypes = testing::Types<Real, Complex, Quaternions, Octonions>;

VectorXd sort(const VectorXd& x) {
  auto y = x;
  std::sort(y.data(), y.data() + x.rows());
  return y;
}

template <typename T>
class TestCases : public testing::Test {
 public:
  void DoMultiplyByIdentity() {
    auto X = T::Random(3, 3);
    auto I = T::Identity(3);
    EXPECT_TRUE(T::IsEqual(X, T::Multiply(X, I)));
  }
  void VerifyJordanIdentity(int n) {
    using Matrix = typename T::Matrix;
    Matrix A = T::Random(n, n);
    A = T::Add(A, T::ConjugateTranspose(A));
    Matrix B = T::Random(n, n);
    B = T::Add(B, T::ConjugateTranspose(B));
    auto W = T::JordanMultiply(A, B);

    EXPECT_TRUE(T::IsHermitian(B));
    EXPECT_TRUE(T::IsHermitian(A));
    EXPECT_TRUE(T::IsHermitian(W));

    // Test Jordan identity.
    auto Asqr = T::JordanMultiply(A, A);
    auto BA = T::JordanMultiply(A, B);
    auto P1 = T::JordanMultiply(Asqr, BA);

    auto BAsqr = T::JordanMultiply(B, Asqr);
    auto P2 = T::JordanMultiply(A, BAsqr);

    EXPECT_TRUE(T::IsEqual(P1, P2));
  }

  void DoQuadraticRepresentationAssociativeTest(int d) {
    using Matrix = typename T::Matrix;
    Matrix A = T::Random(d, d);
    A = T::Add(A, T::ConjugateTranspose(A));
    Matrix B = T::Random(d, d);
    B = T::Add(B, T::ConjugateTranspose(B));
    Matrix Yref = T::QuadraticRepresentation(B, A);

    Matrix Y1 = T::Multiply(T::Multiply(B, A), B);
    Matrix Y2 = T::Multiply(B, T::Multiply(A, B));
    Y1 = T::ScalarMultiply(Y1, .5);
    Y2 = T::ScalarMultiply(Y2, .5);
    Y1 = T::Add(Y1, Y2);

    if (std::is_same<T, Octonions>::value) {
      EXPECT_FALSE(T::IsEqual(Y1, Yref));
    } else {
      EXPECT_TRUE(T::IsEqual(Y1, Yref));
    }
  }

  void DoEigenvalueTests() {
    double eps = 1e-9;
    using Matrix = typename T::Matrix;
    int n = 3;
    Matrix Q = T::Random(n, n);
    Q = T::Add(Q, T::ConjugateTranspose(Q));
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

  void DoTestOrthogonal(int d) {
    if (std::is_same<T, Octonions>::value) {
      return;
    }
    double eps = 1e-8;
    auto Q = T::Random(d, d);

    Q = T::Orthogonalize(Q);

    auto I = T::ScalarMultiply(T::Identity(d), -1);
    auto res = T::Add(I, T::Multiply(Q, T::ConjugateTranspose(Q)));
    EXPECT_TRUE(T::TraceInnerProduct(res, res) < eps);
  }

  void DoEigenvaluesFromSpectralDecomp(int d) {
    if (std::is_same<T, Octonions>::value) {
      return;
    }
    double eps = 1e-8;
    auto Q = T::Random(d, d);

    Q = T::Orthogonalize(Q);
    HyperComplexMatrix D = T::Zero(d, d);
    for (int i = 0; i < d; i++) {
      D.at(0)(i, i) = -d / 2 + i;
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

  void DoAsymmetricEigenvaluesTest(int d) {
    if (std::is_same<T, Octonions>::value) {
      return;
    }
    double eps = 1e-8;
    auto Wsqrt = T::Random(d, d);
    Wsqrt = T::Add(Wsqrt, T::ConjugateTranspose(Wsqrt));
    auto S = T::Random(d, d);
    S = T::Add(S, T::ConjugateTranspose(S));

    auto W = T::Multiply(Wsqrt, T::ConjugateTranspose(Wsqrt));

    VectorXd ref = sort(T::Eigenvalues(T::QuadraticRepresentation(Wsqrt, S)));

    VectorXd calc;
    calc = sort(T::EigenvaluesOfJacobiMatrix(T::Multiply(W, S), W, d));
    auto calc2 = sort(
        T::ApproximateEigenvalues(T::Multiply(W, S), W, T::Random(d, 1), d));
    for (int i = 0; i < d; i++) {
      EXPECT_NEAR(calc(i), calc2(i), eps);
    }

    for (int i = 0; i < d; i++) {
      EXPECT_NEAR(calc(i), ref(i), eps);
    }
  }

  void RankOneTest(int d) {
    assert(d == 3);
    double eps = 1e-5;
    auto Wsqrt = T::Random(d, 1);

    if (std::is_same<T, Octonions>::value) {
      // Lemma 14.90 Spinors and Calibrations By F. Reese Harvey shows
      // that primitive idempotents of the Albert algebra
      // are of the form w w^*, with associator [w_1 w_2 w_3] = 0.
      // This implies w_i are contained in a quaternion subalgebra.
      auto WsqrtQ = Quaternions::Random(d, 1);
      Wsqrt = T::Zero(d, 1);
      for (int i = 0; i < 4; i++) {
        Wsqrt.at(i) = WsqrtQ.at(i);
      }
    }

    Wsqrt = T::ScalarMultiply(Wsqrt, 1.0 / Wsqrt.norm());
    auto W = T::Multiply(Wsqrt, T::ConjugateTranspose(Wsqrt));

    // Add noise to make eigenvalues distinct.
    W.at(0)(0, 0) += 0.00003 * eps;
    W.at(0)(1, 1) += 0.00001 * eps;
    W.at(0)(2, 2) += 0.00002 * eps;

    VectorXd ref(d);
    ref.setZero();
    ref(0) = 1;
    ref = sort(ref);
    VectorXd calc = sort(T::Eigenvalues(W));
    for (int i = 0; i < d; i++) {
      EXPECT_NEAR(calc(i), ref(i), eps);
    }
  }
};

TYPED_TEST_CASE(TestCases, JordanTypes);
TYPED_TEST(TestCases, MultiplyByIdentity) {
  TestFixture::DoMultiplyByIdentity();
}

TYPED_TEST(TestCases, JordanIdentity) { TestFixture::VerifyJordanIdentity(3); }

TYPED_TEST(TestCases, QuadraticRepresentationAssociativeTest) {
  TestFixture::DoQuadraticRepresentationAssociativeTest(3);
}

TYPED_TEST(TestCases, EigenvalueTests) { TestFixture::DoEigenvalueTests(); }

TYPED_TEST(TestCases, DoTestOrthogonal) { TestFixture::DoTestOrthogonal(3); }

TYPED_TEST(TestCases, DoAsymmetricEigenvaluesTest) {
  TestFixture::DoAsymmetricEigenvaluesTest(3);
}

TYPED_TEST(TestCases, DoEigenvaluesFromSpectralDecomp) {
  TestFixture::DoEigenvaluesFromSpectralDecomp(3);
}

TYPED_TEST(TestCases, RankOneTest) { TestFixture::RankOneTest(3); }

TEST(JordanMatrixAlgebra, HermitianRealMatchesEigen) {
  using T = Real;
  int n = 3;
  auto Q = T::Random(n, n);
  Q = T::JordanMultiply(Q, Q);

  EXPECT_TRUE(
      (sort(T::Eigenvalues(Q)) - sort(eig(Q.at(0)).eigenvalues)).norm() < 1e-8);
}

}  // namespace conex


