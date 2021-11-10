#include "conex/supernodal_solver.h"
#include "conex/block_triangular_operations.h"
#include "conex/debug_macros.h"

#include "gtest/gtest.h"
#include <Eigen/Dense>

namespace conex {

std::vector<int> ResidualSize(std::vector<Clique>& path) {
  std::vector<int> y;
  for (size_t j = 0; j < path.size() - 1; j++) {
    std::vector<int> temp;
    IntersectionOfSorted(path.at(j), path.at(j + 1), &temp);
    y.push_back(path.at(j).size() - temp.size());
  }
  y.push_back(path.back().size());
  return y;
}

void RunningIntersectionClosure(std::vector<Clique>* path) {
  if (path->size() < 2) {
    return;
  }
  int n = path->size();
  for (int i = 0; i < n - 2; i++) {
    for (int j = n - 1; j > i + 1; j--) {
      std::vector<int> temp;
      IntersectionOfSorted(path->at(i), path->at(j), &temp);
      if (temp.size() == 0) {
        continue;
      }
      for (int k = j - 1; k > i; k--) {
        path->at(k) = UnionOfSorted(path->at(k), temp);
      }
    }
  }
}

SparseTriangularMatrix MakeSparseTriangularMatrix(
    int N, const std::vector<Clique>& path_) {
  auto path = path_;
  Sort(&path);
  RunningIntersectionClosure(&path);
  auto supernode_size = ResidualSize(path);
  return SparseTriangularMatrix(N, path, supernode_size);
}

SparseTriangularMatrix GetFillInPattern(
    int N, const std::vector<Clique>& cliques_input) {
  auto mat = MakeSparseTriangularMatrix(N, cliques_input);

  for (int j = static_cast<int>(mat.path.size()) - 1; j >= 0; j--) {
    // Initialize columns of super nodes.
    mat.supernodes.at(j).setConstant(1);
    mat.separator.at(j).setConstant(1);

    // Update other columns: the (seperator, seperator) components.
    int index = 0;
    auto s_s = mat.workspace_.seperator_diagonal.at(j);
    int n = mat.path.at(j).size();
    for (int i = mat.supernode_size.at(j); i < n; i++) {
      for (int k = i; k < n; k++) {
        *s_s.at(index++) += 1;
      }
    }
  }
  return mat;
}

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

GTEST_TEST(Basic, Basic) {
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

GTEST_TEST(GetPattern, Basic) {
  vector<Clique> cliques{{0, 1, 2, 5}, {1, 4, 2, 5}, {3, 4, 5}};
}

GTEST_TEST(LowerTri, Constant) {
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

GTEST_TEST(LowerTri, Cholesky) {
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

GTEST_TEST(LowerTri, InverseTest) {
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

GTEST_TEST(LowerTri, InverseOfTranspose) {
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

GTEST_TEST(SupernodalSolver, TestFullSolver) {
  vector<Clique> cliques{{0, 1, 2, 4, 5}, {3, 4}, {5}, {6, 7, 8}};
  auto mat = GetFillInPattern(GetMax(cliques) + 1, cliques);
  for (auto& sn : mat.supernodes) {
    sn.diagonal().array() += 10;
  }
  int val = 0;
  for (size_t i = 0; i < cliques.size(); i++) {
    auto SS = mat.workspace_.seperator_diagonal.at(i);
    Foo data;
    data.supernode_block = mat.supernodes.at(i).data();
    data.separator_supernode_block = mat.separator.at(i).data();
    data.separator_block = SS.data();
    data.num_supernodes = mat.supernodes.at(i).cols();
    data.num_separators = mat.separator.at(i).cols();
    val = Set(val, &data);
  }
}

}  // namespace conex
