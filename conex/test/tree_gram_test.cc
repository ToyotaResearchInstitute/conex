#include "conex/tree_gram.h"
#include "conex/block_triangular_operations.h"

#include <Eigen/Dense>
#include "gtest/gtest.h"

using Eigen::MatrixXd;
using T = TriangularMatrixOperations;
using B = BlockTriangularOperations;

int GetMax(const std::vector<Clique>& cliques) {
  int max = cliques.at(0).at(0);
  for (const auto& c : cliques) {
    for (const auto ci : c) {
      if (ci > max) {
        max = ci;
      }
    }
  }
  return max;
}

MatrixXd GetMatrix(int N, const std::vector<Clique>& c) {
  MatrixXd M(N, N);
  M.setZero();
  for (unsigned int k = 0; k < c.size(); k++) {
    int i = 0;
    for (auto ci : c.at(k)) {
      int j = 0;
      for (auto cj : c.at(k)) {
        M(ci, cj) += 1;
        j++;
      }
      i++;
    }
  }
  return M;
}

MatrixXd GetMatrix(int N, const std::vector<Clique>& c, double val) {
  MatrixXd M(N, N);
  M.setZero();
  for (unsigned int k = 0; k < c.size(); k++) {
    int i = 0;
    for (auto ci : c.at(k)) {
      int j = 0;
      for (auto cj : c.at(k)) {
        M(ci, cj) = val;
        j++;
      }
      i++;
    }
  }
  return M;
}



bool DoPatternTest(const std::vector<Clique>& cliques) {
  int N = GetMax(cliques) + 1;
  MatrixXd error = GetMatrix(N, cliques) - T::ToDense(GetFillInPattern(N, cliques));
  error = error.triangularView<Eigen::Lower>();
  return error.norm() == 0;
}
TEST(Basic, Basic) {
  std::vector<Clique> cliques1{{0, 1, 5},
                              {1, 2, 5},
                              {3, 4, 5}};

  EXPECT_TRUE(DoPatternTest(cliques1));

  std::vector<Clique> cliques2{{0, 1, 2}};
  EXPECT_TRUE(DoPatternTest(cliques2));

  EXPECT_TRUE(DoPatternTest({{0, 1, 2, 4}, {3, 4}, {5, 6, 7}}));
}

std::vector<int> RandomTuple(int max, int size) {
  std::vector<int> y(size);
  for (int i = 0; i < size; i++) {
    y.at(i) = rand() % max;
  }
  return y;
}

// For list of sets A_i, we apply the update:
//
// A_i = A_i \cup (A_{i-1} \cap A_{i+1})
//
//During first application Of RunningIntersectionClosure,
// Things added to A_{i-1} are in A_i.
// Things added to A_{i+1} are in A_i.
// So, on next applicatoin, A_i won't change.
TEST(RunningIntersectionClosureIsIdemponent, Basic) {
  for (int k = 0; k < 10; k++) {
    std::vector<Clique> cliques;
    for (int i = 0; i < 10; i++) {
      cliques.push_back(RandomTuple(10, 3));
    }

    Sort(&cliques);
    RunningIntersectionClosure(&cliques);
    auto cliques_1 = cliques;
    RunningIntersectionClosure(&cliques_1);
    EXPECT_EQ(cliques_1, cliques);
  }
}

TEST(InducedSubtree, Basic) {
// Find spanning tree of clique tree with the induced subtree property.
// Induced subtree property:
//      1) if v* is only in 2 cliques, then cliques must have edge.
//      2) if clique c_i only intersects c_j, then cliques must have edge.
//  std::vector<Clique> cliques{{1, 0, 2, 5},    // v^* = 1,
//                              {0, 2, 3, 4, 5}, // v^* = 5
//                              {3, 4, 5}};
}

TEST(GetPattern, Basic) {
  std::vector<Clique> cliques{{0, 1, 2, 5},
                              {1, 4, 2, 5},
                              {3, 4, 5}};

}

TEST(LowerTri, Constant) {
  using T = TriangularMatrixOperations;
  std::vector<Clique> cliques{{0, 1, 5},
                              {1, 2, 5},
                              {3, 4, 5}};

  auto mat = MakeSparseTriangularMatrix(GetMax(cliques) + 1, cliques);
  T::SetConstant(&mat, -1);
  auto y = T::ToDense(mat);
  auto yref = GetMatrix(GetMax(cliques) + 1, cliques, -1);
  MatrixXd error = y - yref;
  error = error.triangularView<Eigen::Lower>();
  EXPECT_TRUE(error.norm() == 0);
}

void DoCholeskyTest(const std::vector<Clique>& cliques) {
  auto mat = GetFillInPattern(GetMax(cliques) + 1, cliques);
  for (auto& sn : mat.supernodes) {
    sn.diagonal().array() += 100;
  }

  Eigen::MatrixXd x = T::ToDense(mat);
  Eigen::LLT<MatrixXd> llt(x);
  MatrixXd L = llt.matrixL();
  EXPECT_TRUE(llt.info() == Eigen::Success);

  T::CholeskyInPlace(&mat);
  MatrixXd error = T::ToDense(mat) - L;
  error = error.triangularView<Eigen::Lower>();
  EXPECT_NEAR(error.norm(), 0, 1e-12);
}

TEST(LowerTri, Cholesky) {
  DoCholeskyTest({{0, 1, 2},  { 2} });

  DoCholeskyTest({{0, 1, 2, 4}, {3, 4}, {5, 6, 7}});
  DoCholeskyTest({{0, 1, 5},
                  {1, 2, 5},
                  {3, 4, 5}});

  DoCholeskyTest({{0, 1, 2},
                  {1, 2, 3},
                  {3, 4, 2}});

  DoCholeskyTest({{0, 1}, {2, 4}, {3, 4}, {5, 6, 7}, {7, 8, 9, 10 }  });
}

void DoInverseTest(const std::vector<Clique>& cliques) {
  auto mat = GetFillInPattern(GetMax(cliques) + 1, cliques);
  for (auto& sn : mat.supernodes) {
    sn.diagonal().array() += 10;
  }

  Eigen::MatrixXd L = T::ToDense(mat).triangularView<Eigen::Lower>();
  Eigen::VectorXd b;
  b.setLinSpaced(L.rows(), -1, 1);
  auto y = T::ApplyInverse(&mat, b);
  EXPECT_NEAR((L*y - b).norm(), 0, 1e-12);

}

TEST(LowerTri, InverseTest) {
  DoInverseTest({{0, 1, 2, 3}, {3, 4, 5}});
  DoInverseTest({{0, 1, 2, 3}});
  DoInverseTest({{0, 1, 2, 3}, {3, 4}, {4, 5, 6}});
}

void DoInverseOfTransposeTest(const std::vector<Clique>& cliques) {
  auto mat = GetFillInPattern(GetMax(cliques) + 1, cliques);
  for (auto& sn : mat.supernodes) {
    sn.diagonal().array() += 10;
  }

  Eigen::MatrixXd L = T::ToDense(mat).triangularView<Eigen::Lower>();
  Eigen::VectorXd b;
  b.setLinSpaced(L.rows(), -1, 1);
  auto y = T::ApplyInverseOfTranspose(&mat, b);
  EXPECT_NEAR((L.transpose()*y - b).norm(), 0, 1e-12);


}

TEST(LowerTri, InverseOfTranspose) {
  DoInverseOfTransposeTest({{0, 1, 2, 5}, {3, 4, 5}});
  DoInverseOfTransposeTest({{0, 1, 2, 5}, {3, 4, 5}, {5, 6}});
  DoInverseOfTransposeTest({{0, 1, 2, 3}});
}
