#pragma once
#include <array>
#include <stack>
#include <vector>
#include "debug_macros.h"

using std::array;
using std::vector;
constexpr int N = 5;

namespace conex {
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

vector<int> path(int x, int y, const std::vector<int>& tree,
                 const std::vector<int>& height);

struct RootedTree {
  RootedTree(int number_of_nodes)
      : parent(number_of_nodes), height(number_of_nodes) {}
  std::vector<int> parent;
  std::vector<int> height;
};

template <typename T>
RootedTree BuildRootTree(const T& A, int root) {
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
    int parent = node_stack.top();
    node_stack.pop();
    auto argmin = ArgMax(A[parent], tree);
    for (auto i : argmin) {
      tree[i] = parent;
      height[i] = height[parent] + 1;
      node_stack.push(i);
    }
  }
  return rooted_tree;
}

}  // namespace conex
