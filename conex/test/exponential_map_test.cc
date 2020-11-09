#include "conex/exponential_map.h"

#include "gtest/gtest.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
using Eigen::MatrixXd;
using Eigen::VectorXd;

using JordanTypes = testing::Types<Real, Complex, Quaternions>;

template<typename T>
HyperComplexMatrix RandomOrthogonal(int n) {
  return T::Orthogonalize(T::Random(n, n));
}

template<typename T>
HyperComplexMatrix BuildMatrix(const VectorXd& eigenvalues, const HyperComplexMatrix& Q) {
  int n = eigenvalues.rows();
  assert(n = Q.at(0).rows());
  auto D = T::Zero(n, n); 
  D.at(0).diagonal() = eigenvalues;
  return T::Multiply(T::Multiply(Q,  D), T::ConjugateTranspose(Q));
}

TEST(MatrixExponential, CompareWithEigen) {
  int n = 4;
  MatrixXd A(n, n); 
  A << 3, 1, 0, 1,
       1, 3, 1, 0,
       0, 1, 4, 1,
       1, 0, 1, 5;
  A = A * .001;

  MatrixXd reference = A.exp();
  MatrixXd calculated(n, n);
      
  HyperComplexMatrix arg(1); arg.at(0) = A;
  HyperComplexMatrix result(1);
  ExponentialMap(arg, &result);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      EXPECT_NEAR(reference(i, j), result.at(0)(i, j), 1e-7);
    }
  }
}

template<typename T>
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
    for (int k = 0; k < result.size(); k++) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          EXPECT_NEAR(reference.at(k)(i, j), result.at(k)(i, j), 1e-7);
        }
      }
    }
  }
};

TYPED_TEST_CASE(TestCases, JordanTypes);
TYPED_TEST(TestCases, MultiplyByIdentity) {
  TestFixture::CompareWithReference();
}

