#include "conex/supernodal_solver.h"
#include "conex/block_triangular_operations.h"
#include "conex/debug_macros.h"

#include "gtest/gtest.h"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using T = TriangularMatrixOperations;
using B = BlockTriangularOperations;
using std::vector;

int GetMax(const vector<Clique>& cliques) {
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

MatrixXd GetMatrix(int N, const vector<Clique>& c) {
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

MatrixXd GetMatrix(int N, const vector<Clique>& c, double val) {
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

bool DoPatternTest(const vector<Clique>& cliques) {
  int N = GetMax(cliques) + 1;
  MatrixXd error =
      GetMatrix(N, cliques) - T::ToDense(GetFillInPattern(N, cliques));
  error = error.triangularView<Eigen::Lower>();
  return error.norm() == 0;
}

TEST(Basic, Basic) {
  vector<Clique> cliques1{{0, 1, 5}, {1, 2, 5}, {3, 4, 5}};

  EXPECT_TRUE(DoPatternTest(cliques1));

  vector<Clique> cliques2{{0, 1, 2}};
  EXPECT_TRUE(DoPatternTest(cliques2));

  EXPECT_TRUE(DoPatternTest({{0, 1, 2, 4}, {3, 4}, {5, 6, 7}}));
}

vector<int> RandomTuple(int max, int size) {
  vector<int> y(size);
  for (int i = 0; i < size; i++) {
    y.at(i) = rand() % max;
  }
  return y;
}

// For list of sets A_i, we apply the update:
//
// A_i = A_i \cup (A_{i-1} \cap A_{i+1})
//
// During first application Of RunningIntersectionClosure,
// Things added to A_{i-1} are in A_i.
// Things added to A_{i+1} are in A_i.
// So, on next applicatoin, A_i won't change.
TEST(RunningIntersectionClosureIsIdemponent, Basic) {
  for (int k = 0; k < 10; k++) {
    vector<Clique> cliques;
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

TEST(GetPattern, Basic) {
  vector<Clique> cliques{{0, 1, 2, 5}, {1, 4, 2, 5}, {3, 4, 5}};
}

TEST(LowerTri, Constant) {
  using T = TriangularMatrixOperations;
  vector<Clique> cliques{{0, 1, 5}, {1, 2, 5}, {3, 4, 5}};

  auto mat = MakeSparseTriangularMatrix(GetMax(cliques) + 1, cliques);
  T::SetConstant(&mat, -1);
  auto y = T::ToDense(mat);
  auto yref = GetMatrix(GetMax(cliques) + 1, cliques, -1);
  MatrixXd error = y - yref;
  error = error.triangularView<Eigen::Lower>();
  EXPECT_TRUE(error.norm() == 0);
}

void DoCholeskyTest(const vector<Clique>& cliques) {
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
  DoCholeskyTest({{0, 1, 2}, {2}});

  DoCholeskyTest({{0, 1, 2, 4}, {3, 4}, {5, 6, 7}});
  DoCholeskyTest({{0, 1, 5}, {1, 2, 5}, {3, 4, 5}});

  DoCholeskyTest({{0, 1, 2}, {1, 2, 3}, {3, 4, 2}});

  DoCholeskyTest({{0, 1}, {2, 4}, {3, 4}, {5, 6, 7}, {7, 8, 9, 10}});
}

void DoInverseTest(const vector<Clique>& cliques) {
  auto mat = GetFillInPattern(GetMax(cliques) + 1, cliques);
  for (auto& sn : mat.supernodes) {
    sn.diagonal().array() += 10;
  }

  Eigen::MatrixXd L = T::ToDense(mat).triangularView<Eigen::Lower>();
  Eigen::VectorXd b;
  b.setLinSpaced(L.rows(), -1, 1);
  auto y = T::ApplyInverse(&mat, b);
  EXPECT_NEAR((L * y - b).norm(), 0, 1e-12);
}

TEST(LowerTri, InverseTest) {
  DoInverseTest({{0, 1, 2, 3}, {3, 4, 5}});
  DoInverseTest({{0, 1, 2, 3}});
  DoInverseTest({{0, 1, 2, 3}, {3, 4}, {4, 5, 6}});
}

void DoInverseOfTransposeTest(const vector<Clique>& cliques) {
  auto mat = GetFillInPattern(GetMax(cliques) + 1, cliques);
  for (auto& sn : mat.supernodes) {
    sn.diagonal().array() += 10;
  }

  Eigen::MatrixXd L = T::ToDense(mat).triangularView<Eigen::Lower>();
  Eigen::VectorXd b;
  b.setLinSpaced(L.rows(), -1, 1);
  auto y = T::ApplyInverseOfTranspose(&mat, b);
  EXPECT_NEAR((L.transpose() * y - b).norm(), 0, 1e-12);
}

TEST(LowerTri, InverseOfTranspose) {
  DoInverseOfTransposeTest({{0, 1, 2, 5}, {3, 4, 5}});
  DoInverseOfTransposeTest({{0, 1, 2, 5}, {3, 4, 5}, {5, 6}});
  DoInverseOfTransposeTest({{0, 1, 2, 3}});
}

typedef struct Foo {
  double* supernode_block;
  double* separator_supernode_block;
  int num_supernodes;
  int num_separators;
  double** separator_block;
  int seperator_block_stride = -1;
} Foo;

int Set(int initial_value, Foo* data) {
  int cnt = initial_value;
  int num_n = data->num_supernodes;
  int num_s = data->num_separators;
  double* n = data->supernode_block;
  for (int j = 0; j < num_n; j++) {
    for (int i = j; i < num_n; i++) {
      n[i + j * num_n] = cnt++;
    }
  }

  double* s_n = data->separator_supernode_block;
  for (int j = 0; j < num_s; j++) {
    for (int i = 0; i < num_n; i++) {
      s_n[i + j * num_n] = cnt++;
    }
  }

  int index = 0;
  for (int j = 0; j < num_s; j++) {
    for (int i = j; i < num_s; i++) {
      *data->separator_block[index++] = cnt++;
    }
  }
  return cnt;
}

TEST(SupernodalSolver, TestFullSolver) {
  vector<Clique> cliques{{0, 1, 2, 4, 5}, {3, 4}, {5}, {6, 7, 8}};
  auto mat = GetFillInPattern(GetMax(cliques) + 1, cliques);
  for (auto& sn : mat.supernodes) {
    sn.diagonal().array() += 10;
  }
  DUMP(T::ToDense(mat));
  int val = 0;
  for (int i = 0; i < cliques.size(); i++) {
    auto SS = mat.workspace_.seperator_diagonal.at(i);
    Foo data;
    data.supernode_block = mat.supernodes.at(i).data();
    data.separator_supernode_block = mat.separator.at(i).data();
    data.separator_block = SS.data();
    data.num_supernodes = mat.supernodes.at(i).cols();
    data.num_separators = mat.separator.at(i).cols();
    val = Set(val, &data);
  }
  DUMP(T::ToDense(mat));
}
