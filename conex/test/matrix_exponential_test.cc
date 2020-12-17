#include "gtest/gtest.h"

#include "conex/matrix_exponential.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "conex/debug_macros.h"

namespace conex {

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(MatrixExponential, CompareWithEigen) {
  int n = 4;
  MatrixXd A(n, n);
  // clang-format off
  A << 3, 1, 0, 1,
       1, 3, 1, 0,
       0, 1, 4, 1,
       1, 0, 1, 5;
  // clang-format on
  A = A / A.trace();

  MatrixXd reference = A.exp();
  MatrixXd calculated(n, n);

  Map<MatrixXd, Eigen::Aligned> map(calculated.data(), n, n);
  MatrixExponential(A, &map);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      EXPECT_NEAR(reference(i, j), calculated(i, j), 1e-7);
    }
  }
}

}  // namespace conex
