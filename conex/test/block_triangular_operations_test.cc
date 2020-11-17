#include "conex/block_triangular_operations.h" 
#include "conex/tree_gram.h"

#include "gtest/gtest.h"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using T = TriangularMatrixOperations;
using B = BlockTriangularOperations;

int GetMax(const std::vector<Clique> &cliques) {
  int max = cliques.at(0).at(0);
  for (const auto &c : cliques) {
    for (const auto ci : c) {
      if (ci > max) {
        max = ci;
      }
    }
  }
  return max;
}

void DoCholeskyTest(const std::vector<Clique> &cliques) {
  auto mat = GetFillInPattern(GetMax(cliques) + 1, cliques);
  for (auto &sn : mat.supernodes) {
    sn.diagonal().array() += 100;
  }
  auto mat2 = mat;

  Eigen::MatrixXd x = T::ToDense(mat);
  Eigen::LLT<MatrixXd> llt(x);
  MatrixXd L = llt.matrixL();
  EXPECT_TRUE(llt.info() == Eigen::Success);

  B::BlockCholeskyInPlace(&mat2);
  MatrixXd error = T::ToDense(mat2) - L;
  error = error.triangularView<Eigen::Lower>();
  EXPECT_NEAR(error.norm(), 0, 1e-12);
}

TEST(LowerTri, Cholesky) {
  DoCholeskyTest({{0, 1, 2}, {2}});
  DoCholeskyTest({{0, 1, 2, 4}, {3, 4}, {5, 6, 7}});
  DoCholeskyTest({{0, 1, 5}, {1, 2, 5}, {3, 4, 5}});
  DoCholeskyTest({{0, 1, 2}, {1, 2, 3}, {3, 4, 2}});
  DoCholeskyTest({{0, 1}, {2, 4}, {3, 4}, {5, 6, 7}, {7, 8, 9, 10}});
}

void DoInverseTest(const std::vector<Clique> &cliques) {
  auto mat = GetFillInPattern(GetMax(cliques) + 1, cliques);
  for (auto &sn : mat.supernodes) {
    sn.diagonal().array() += 10;
  }

  Eigen::MatrixXd L = T::ToDense(mat).triangularView<Eigen::Lower>();
  Eigen::VectorXd b;
  b.setLinSpaced(L.rows(), -1, 1);

  Eigen::VectorXd y2 = b;
  B::ApplyBlockInverseInPlace(mat, &y2);
  EXPECT_NEAR((L * y2 - b).norm(), 0, 1e-12);
}

TEST(LowerTri, InverseTest) {
  DoInverseTest({{0, 1, 2, 3}, {3, 4, 5}});
  DoInverseTest({{0, 1, 2, 3}});
  DoInverseTest({{0, 1, 2, 3}, {3, 4}, {4, 5, 6}});
}

void DoInverseOfTransposeTest(const std::vector<Clique> &cliques) {
  auto mat = GetFillInPattern(GetMax(cliques) + 1, cliques);
  for (auto &sn : mat.supernodes) {
    sn.diagonal().array() += 10;
  }

  Eigen::MatrixXd L = T::ToDense(mat).triangularView<Eigen::Lower>();
  Eigen::VectorXd b;
  b.setLinSpaced(L.rows(), -1, 1);

  Eigen::VectorXd y2 = b;
  B::ApplyBlockInverseOfTransposeInPlace(mat, &y2);
  EXPECT_NEAR((L.transpose() * y2 - b).norm(), 0, 1e-12);
}

TEST(LowerTri, InverseOfTranspose) {
  DoInverseOfTransposeTest({{0, 1, 2, 5}, {3, 4, 5}});
  DoInverseOfTransposeTest({{0, 1, 2, 5}, {3, 4, 5}, {5, 6}});
  DoInverseOfTransposeTest({{0, 1, 2, 3}});
}
