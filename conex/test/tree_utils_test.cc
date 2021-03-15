#include "conex/tree_utils.h"
#include "conex/debug_macros.h"
#include "gtest/gtest.h"

namespace conex {
using std::vector;

inline vector<int> ArgMax(const array<int, N>& x, const vector<int>& tree) {
  vector<int> argmax;
  int val = 1;
  for (size_t i = 0; i < x.size(); i++) {
    if (x[i] >= val && tree[i] == -1) {
      if (x[i] > val) {
        argmax.clear();
        val = x[i];
      }
      argmax.push_back(i);
    }
  }
  return argmax;
}

template <typename T>
RootedTree GetSpanningTree(const T& adjacency_matrix, int root) {
  RootedTree rooted_tree(N);
  auto& tree = rooted_tree.parent;
  auto& height = rooted_tree.height;
  for (int i = 0; i < N; i++) {
    tree[i] = -1;
    height[i] = 0;
  }

  int parent = root;
  tree[parent] = parent;
  height[parent] = 0;

  std::stack<size_t> node_stack;
  node_stack.push(parent);
  while (node_stack.size() != 0) {
    parent = node_stack.top();
    node_stack.pop();
    auto argmin = ArgMax(adjacency_matrix[parent], tree);
    for (auto i : argmin) {
      tree[i] = parent;
      height[i] = height[parent] + 1;
      node_stack.push(i);
    }
  }
  return rooted_tree;
}

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

GTEST_TEST(TreeUtils, TestPath1) {
  auto A = TestGraph();
  int root = 0;
  auto tree = GetSpanningTree(A, root);
  auto p1 = PathInTree(4, 0, tree.parent, tree.height);
  vector<int> path_ref{4, 3, 0};
  EXPECT_EQ(p1, path_ref);

  path_ref = vector<int>{4, 2, 3, 1, 0};
  auto p2 = PathInTree(4, 2, tree.parent, tree.height);
  EXPECT_EQ(p2, path_ref);
}

GTEST_TEST(TreeUtils, TestPath2) {
  auto A = TestGraph();
  int root = 4;
  auto tree = GetSpanningTree(A, root);
  auto p1 = PathInTree(0, 4, tree.parent, tree.height);
  vector<int> path_ref{0, 3, 4};
  EXPECT_EQ(p1, path_ref);

  path_ref = vector<int>{2, 1, 0, 3, 4};
  auto p2 = PathInTree(2, 4, tree.parent, tree.height);
  EXPECT_EQ(p2, path_ref);
}

}  // namespace conex
