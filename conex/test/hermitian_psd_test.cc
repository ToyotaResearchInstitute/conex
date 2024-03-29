#include "conex/hermitian_psd.h"
#include "conex/dense_lmi_constraint.h"

#include "gtest/gtest.h"
#include <Eigen/Dense>

#include "conex/cone_program.h"

namespace conex {

using JordanTypes = testing::Types<Real, Complex, Quaternions, Octonions>;

using Eigen::MatrixXd;

MatrixXd ToMat(const Real::Matrix& x) { return x.at(0); }

std::vector<MatrixXd> ToMat(const std::vector<Real::Matrix>& x) {
  std::vector<MatrixXd> y;
  for (const auto& e : x) {
    y.push_back(ToMat(e));
  }
  return y;
}
int CompareRealHermitianWithLMI(int rank, int dim) {
  using T = Real;
  using Matrix = typename T::Matrix;
  SolverConfiguration config;
  config.inv_sqrt_mu_max = std::sqrt(1.0 / 1e-4);
  config.final_centering_tolerance = 1e-8;
  config.prepare_dual_variables = true;
  int m = dim;
  std::vector<Matrix> constraint_matrices(m);
  Matrix constraint_affine = T::Identity(rank);
  for (int i = 0; i < m; i++) {
    constraint_matrices.at(i) = T::Random(rank, rank);
    constraint_matrices.at(i) =
        T::Add(constraint_matrices.at(i),
               T::ConjugateTranspose(constraint_matrices.at(i)));
  }
  HermitianPsdConstraint<T> T2(rank, constraint_matrices, constraint_affine);

  Program prog(m);
  DenseMatrix y(m, 1);
  prog.AddConstraint(T2);

  auto b = GetFeasibleObjective(&prog);
  bool solved_1 = Solve(b, prog, config, y.data());
  MatrixXd X_1(rank, rank);
  prog.GetDualVariable(0, &X_1);

  Program prog2(m);
  DenseMatrix y2(m, 1);
  prog2.AddConstraint(DenseLMIConstraint(rank, ToMat(constraint_matrices),
                                         ToMat(constraint_affine)));

  bool solved_2 = Solve(b, prog2, config, y2.data());
  MatrixXd X_2(rank, rank);
  prog2.GetDualVariable(0, &X_2);
  EXPECT_NEAR((y2 - y).norm(), 0, 1e-11);
  EXPECT_NEAR((X_2 - X_1).norm(), 0, 1e-11);

  return solved_1 && solved_2;
}

template <typename T>
class TestCases : public testing::Test {
 public:
  using Type = T;

  void DoSolve(int rank, int m) {
    using Matrix = typename T::Matrix;
    SolverConfiguration config;

    config.inv_sqrt_mu_max = 1000;
    config.final_centering_steps = 4;
    config.max_iterations = 100;

    std::vector<Matrix> constraint_matrices(m);
    Matrix constraint_affine = T::Identity(rank);
    for (int i = 0; i < m; i++) {
      constraint_matrices.at(i) = T::Random(rank, rank);
      constraint_matrices.at(i) =
          T::Add(constraint_matrices.at(i),
                 T::ConjugateTranspose(constraint_matrices.at(i)));
    }
    HermitianPsdConstraint<T> T2(rank, constraint_matrices, constraint_affine);

    Program prog(m);
    DenseMatrix y(m, 1);
    prog.AddConstraint(T2);

    auto b = GetFeasibleObjective(&prog);
    EXPECT_TRUE(Solve(b, prog, config, y.data()));
  }
  void SolveRandomInstances(int n, int m) {
    for (int i = 0; i < 1; i++) {
      DoSolve(n, m);
    }
  }
};

TYPED_TEST_CASE(TestCases, JordanTypes);
TYPED_TEST(TestCases, SolveRandomInstances) {
  for (int i = 0; i < 3; i++) {
    if (!std::is_same<typename TestFixture::Type, Octonions>::value) {
      TestFixture::SolveRandomInstances(3 + i * 40, 2 + 3 * i);
    } else {
      TestFixture::SolveRandomInstances(3, 2 + 3 * i);
    }
  }
}

GTEST_TEST(Hermitian, CompareWithLMI) {
  srand(0);
  for (int i = 0; i < 4; i++) {
    int num_vars = 4;
    int rank = 8;
    EXPECT_TRUE(CompareRealHermitianWithLMI(rank, num_vars));
  }
}

}  // namespace conex
