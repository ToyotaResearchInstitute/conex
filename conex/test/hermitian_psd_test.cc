#include "conex/hermitian_psd.h"
#include "conex/dense_lmi_constraint.h"

#include "gtest/gtest.h"
#include <Eigen/Dense>

#include "conex/cone_program.h"

using JordanTypes = testing::Types<Real, Complex, Quaternions, Octonions>;

using Eigen::MatrixXd;

MatrixXd ToMat(const Real::Matrix& x) {
  return x.at(0);
}

std::vector<MatrixXd> ToMat(const std::vector<Real::Matrix>& x) {
  std::vector<MatrixXd> y;
  for (const auto& e : x) {
    y.push_back(ToMat(e));
  }
  return y;
}
int CompareRealHermitianWithLMI() {
  using T = Real;
  using Matrix = typename T::Matrix;
  SolverConfiguration config;
  config.inv_sqrt_mu_max = std::sqrt(1.0/1e-4);
  int m = 2;
  int rank = 3;
  std::vector<Matrix> constraint_matrices(m);
  Matrix constraint_affine = T::Identity(rank);
  for (int i = 0; i < m; i++) {
    constraint_matrices.at(i) = T::Random(rank, rank);
    constraint_matrices.at(i) = T::Add(constraint_matrices.at(i), 
                                       T::ConjugateTranspose(constraint_matrices.at(i)));
  }
  HermitianPsdConstraint<T> T2(3, constraint_matrices, constraint_affine);

  Program prog;
  DenseMatrix y(m, 1);
  prog.constraints.push_back(T2);

  auto b = GetFeasibleObjective(m, prog.constraints);
  bool solved_1 = Solve(b, prog, config, y.data());

  Program prog2;
  DenseMatrix y2(m, 1);
  prog2.constraints.push_back(DenseLMIConstraint(3, ToMat(constraint_matrices),
                                                 ToMat(constraint_affine)));

  bool solved_2 = Solve(b, prog2, config, y2.data());
  EXPECT_TRUE((y2 - y).norm() < 1e-5);

  return solved_1 && solved_2;
}

template<typename T>
class TestCases : public testing::Test {
 public:
  void DoSolve() {
    using Matrix = typename T::Matrix;
    SolverConfiguration config;
    
    config.inv_sqrt_mu_max = 1000;
    config.final_centering_steps = 4;
    config.max_iterations = 100;

    int m = 2;
    int rank = 3;
    std::vector<Matrix> constraint_matrices(m);
    Matrix constraint_affine = T::Identity(rank);
    for (int i = 0; i < m; i++) {
      constraint_matrices.at(i) = T::Random(rank, rank); constraint_matrices.at(i) = T::Add(constraint_matrices.at(i), T::ConjugateTranspose(constraint_matrices.at(i)));
    }
    HermitianPsdConstraint<T> T2(3, constraint_matrices, constraint_affine);

    Program prog;
    DenseMatrix y(m, 1);
    prog.constraints.push_back(T2);

    auto b = GetFeasibleObjective(m, prog.constraints);
    EXPECT_TRUE(Solve(b, prog, config, y.data()));
  }
  void SolveRandomInstances() {
    for (int i = 0; i < 1; i++) {
      DoSolve();
    }
  }
};

TYPED_TEST_CASE(TestCases, JordanTypes);
TYPED_TEST(TestCases, SolveRandomInstances) {
  TestFixture::SolveRandomInstances();
}
TEST(Hermitian, CompareWithLMI) {
  for (int i = 0; i < 2; i++) {
    EXPECT_TRUE(CompareRealHermitianWithLMI());
  }
}
