#include "conex/tree_utils.h"
#include "conex/debug_macros.h"
#include "gtest/gtest.h"

namespace conex {
using std::vector;

array<array<int, N>, N> TestGraph() {
  // Graph
  //
  //        0
  //     1     3
  //     2     4

  array<array<int, N>, N> A;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = 0;
    }
  }

  A[0][1] = 1;
  A[1][2] = 1;

  A[0][3] = 1;
  A[3][4] = 1;

  for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
      A[j][i] = A[i][j];
    }
  }
  return A;
}

TEST(TreeUtils, TestPath1) {
  auto A = TestGraph();
  int root = 0;
  auto tree = BuildRootTree(A, root);
  auto p1 = path(4, 0, tree.parent, tree.height);
  vector<int> path_ref{4, 3, 0};
  EXPECT_EQ(p1, path_ref);

  path_ref = vector<int>{4, 2, 3, 1, 0};
  auto p2 = path(4, 2, tree.parent, tree.height);
  EXPECT_EQ(p2, path_ref);
}

TEST(TreeUtils, TestPath2) {
  auto A = TestGraph();
  int root = 4;
  auto tree = BuildRootTree(A, root);
  auto p1 = path(0, 4, tree.parent, tree.height);
  vector<int> path_ref{0, 3, 4};
  EXPECT_EQ(p1, path_ref);

  path_ref = vector<int>{2, 1, 0, 3, 4};
  auto p2 = path(2, 4, tree.parent, tree.height);
  EXPECT_EQ(p2, path_ref);
}

}  // namespace conex
