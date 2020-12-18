#include "conex/exponential_map.h"

#include "conex/jordan_matrix_algebra.h"
#include "gtest/gtest.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace conex {



using Eigen::MatrixXd;
using Eigen::VectorXd;

using JordanTypes = testing::Types<Real, Complex, Quaternions>;

template <typename T>
HyperComplexMatrix RandomOrthogonal(int n) {
  return T::Orthogonalize(T::Random(n, n));
}

template <typename T>
HyperComplexMatrix BuildMatrix(const VectorXd& eigenvalues,
                               const HyperComplexMatrix& Q) {
  int n = eigenvalues.rows();
  assert(n = Q.at(0).rows());
  auto D = T::Zero(n, n);
  D.at(0).diagonal() = eigenvalues;
  return T::Multiply(T::Multiply(Q, D), T::ConjugateTranspose(Q));
}

TEST(ExponentialMapPadeApproximation, CompareWithEigen) {
  int n = 4;
  MatrixXd A(n, n);
  A << 3, 1, 0, 1, 1, 3, 1, 0, 0, 1, 4, 1, 1, 0, 1, 5;
  A = A * .001;

  MatrixXd reference = A.exp();
  MatrixXd calculated(n, n);

  HyperComplexMatrix arg(1);
  arg.at(0) = A;
  HyperComplexMatrix result(1);
  ExponentialMap(arg, &result);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      EXPECT_NEAR(reference(i, j), result.at(0)(i, j), 1e-7);
    }
  }
}

template <typename T>
class TestCases : public testing::Test {
 public:
  void CompareWithReference() {
    int n = 4;
    VectorXd eigenvalues(n);
    eigenvalues << -.1, .1, .1, .01;
    auto Q = RandomOrthogonal<T>(n);
    auto arg = BuildMatrix<T>(eigenvalues, Q);
    auto reference = BuildMatrix<T>(eigenvalues.array().exp(), Q);

    auto result = T::Zero(n, n);
    ExponentialMap(arg, &result);
    for (size_t k = 0; k < result.size(); k++) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          //TODO(FrankPermenter): reduce this threshold.
          EXPECT_NEAR(reference.at(k)(i, j), result.at(k)(i, j), 1e-4);
        }
      }
    }
  }
};

TYPED_TEST_CASE(TestCases, JordanTypes);
TYPED_TEST(TestCases, MultiplyByIdentity) {
  TestFixture::CompareWithReference();
}

template <typename T>
typename T::Matrix Symmetrize(const typename T::Matrix& x) {
  auto y = T::Add(x, T::ConjugateTranspose(x));
  y = T::ScalarMultiply(y, .5);
  return y;
}

TEST(TestCases, GeodesicUpdateOctonions) {
  using T = Octonions;
  int order = 3;
  auto s = Symmetrize<T>(T::Random(order, order));
  auto w = T::Identity(order);
  s = T::Add(w, T::ScalarMultiply(s, .1));

  auto y = GeodesicUpdate(w, s);
  EXPECT_TRUE(
      (VectorXd(T::Eigenvalues(s).array().exp()) - T::Eigenvalues(y)).norm() <
      1e-3);

  auto d = T::Zero(order, order);
  for (int i = 0; i < 3; i++) {
    d.at(0)(i, i) = 1 + i * .02;
  }
  y = GeodesicUpdate(T::Identity(order), d);
  for (int i = 0; i < order; i++) {
    EXPECT_NEAR(y.at(0)(i, i), d.at(0).diagonal().array().exp()(i), 1e-4);
  }
}

VectorXd sort(const VectorXd& x) {
  auto y = x;
  std::sort(y.data(), y.data() + x.rows());
  return y;
}

TEST(TestCases, GeodesicUpdateRescaling) {
  using T = Octonions;
  int order = 3;
  auto wsqrt = Symmetrize<T>(T::Random(order, order));
  // auto w = T::Multiply(wsqrt, wsqrt);
  auto w = T::Identity(order);
  auto s = Symmetrize<T>(T::Random(order, order));
  s = T::Add(T::Identity(order), T::ScalarMultiply(s, .05));
  s = T::ScalarMultiply(s, -1);

  auto yref = T::ScalarMultiply(GeodesicUpdate(w, s), std::exp(1));
  auto ycalc = GeodesicUpdateScaled(w, s);

  auto eig_ref = T::Eigenvalues(yref);
  auto eig_calc = T::Eigenvalues(ycalc);
  for (int i = 0; i < order; i++) {
    EXPECT_TRUE(eig_calc(i) >= 0);
    EXPECT_NEAR(eig_ref(i), eig_calc(i), 1e-2);
  }
}

} // namespace conex

