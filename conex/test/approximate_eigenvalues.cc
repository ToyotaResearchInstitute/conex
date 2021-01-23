#include "conex/approximate_eigenvalues.h"

#include <chrono>

#include "gtest/gtest.h"

#include "conex/debug_macros.h"
#include "conex/test/test_util.h"
#include <Eigen/Dense>

namespace conex {

using Eigen::MatrixXd;
using Eigen::VectorXd;

GTEST_TEST(Eigenvalues, NonsymmetricFromJacobiIterations) {
  int n = 4;
  MatrixXd A(n, n);
  A << 3, 1, 0, 1, 1, 3, 1, 0, 0, 1, 4, 1, 1, 0, 1, 5;
  A = A / A.trace();

  MatrixXd W = MatrixXd::Random(n, n);
  W = W * W.transpose();
  A = W * A;

  VectorXd r0(n);
  r0 << 1, 2, 0, 4;
  VectorXd eigJ = ApproximateEigenvalues(A, W, r0, n, false /*no compressed*/);
  std::sort(eigJ.data(), eigJ.data() + eigJ.rows());

  VectorXd eigL = ApproximateEigenvalues(A, W, r0, n, true /*use compressed*/);
  std::sort(eigL.data(), eigL.data() + eigL.rows());
  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(eigL(i), eigJ(i), 1e-12);
  }

  auto eigA = eig(A).eigenvalues;
  std::sort(eigA.data(), eigA.data() + n);
  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(eigJ(i), eigA(i), 1e-12);
  }
}

GTEST_TEST(Eigenvalues, TruncatedApproximiationInterlaces) {
  int n = 4;
  MatrixXd A(n, n);
  A << .1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5;

  VectorXd r0(n);
  r0 << 1, 2, 0, 4;

  int n_approx = 2;
  VectorXd eigJ =
      ApproximateEigenvalues(A, MatrixXd::Identity(n, n), r0, n_approx, true);
  VectorXd eigA = eig(A).eigenvalues;

  std::sort(eigJ.data(), eigJ.data() + eigJ.rows());
  std::sort(eigA.data(), eigA.data() + eigA.rows());
  EXPECT_TRUE(eigJ.tail(1)(0) <= eigA.tail(1)(0));
  EXPECT_TRUE(eigJ.head(1)(0) >= eigA.head(1)(0));
}

GTEST_TEST(Eigenvalues, Lanczos) {
  int n = 4;
  MatrixXd A(n, n);
  A << 3, 1, 0, 1, 1, 3, 1, 0, 0, 1, 4, 1, 1, 0, 1, 5;
  A = A / A.trace();

  VectorXd r0(n);
  r0 << 1, 2, 0, 4;
  VectorXd eigJ =
      ApproximateEigenvalues(A, MatrixXd::Identity(n, n), r0, n, true);
  auto eigA = eig(A).eigenvalues;

  auto eigL = ApproximateEigenvalues(A, r0, n);

  std::sort(eigJ.data(), eigJ.data() + n);
  std::sort(eigA.data(), eigA.data() + n);
  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(eigJ(i), eigA(i), 1e-12);
  }
  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(eigL(i), eigA(i), 1e-12);
  }
}

GTEST_TEST(Eigenvalues, Profile) {
  for (int k = 0; k < 4; k++) {
    int n = 25;
    MatrixXd S = MatrixXd::Random(n, n);
    MatrixXd St = S.transpose();
    S = S + St;
    MatrixXd W = MatrixXd::Random(n, n);
    W = W * W.transpose();
    MatrixXd WS = W * S;
    VectorXd r0 = VectorXd::Random(n);

    auto t1 = std::chrono::high_resolution_clock::now();
    VectorXd eigJ = ApproximateEigenvalues(WS, W, r0, n / 2, true);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration1 =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    VectorXd eigWS = eig(WS).eigenvalues;
    t2 = std::chrono::high_resolution_clock::now();
    auto duration2 =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    EXPECT_LE(duration1, duration2);
    EXPECT_NEAR(eigWS.maxCoeff() / eigJ.maxCoeff(), 1, 1e-2);
  }
}

}  // namespace conex
