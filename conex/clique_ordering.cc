#include "conex/clique_ordering.h"

#include <algorithm>
#include <stack>

#include "conex/debug_macros.h"
#include "conex/supernodal_solver.h"

namespace conex {

using std::vector;
using Cliques = vector<vector<int>>;

namespace {

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

int LinearIndex(int i, int j, int n) {
  if (i > j) {
    return j * n + i;
  } else {
    return i * n + j;
  }
}

int GetUnvisited(const std::vector<int>& x) {
  int cnt = 0;
  for (auto xi : x) {
    if (xi == 0) {
      return cnt;
    }
    cnt++;
  }
  return -1;
}

template <typename T>
class SymmetricMatrix {
 public:
  SymmetricMatrix(int n) : n_(n), data_(n * n) {}
  vector<int>& operator()(int a, int b) {
    return data_.at(LinearIndex(a, b, n_));
  }
  const vector<int>& operator()(int a, int b) const {
    return data_.at(LinearIndex(a, b, n_));
  }
  int n_;
  vector<T> data_;
};

using Edge = std::pair<int, int>;
vector<int> GetMaxWeightedDegreeNode(
    int n, const vector<Edge>& edges,
    const SymmetricMatrix<vector<int>>& intersections) {
  vector<int> weights(n);
  for (auto& w : weights) {
    w = 0;
  }

  for (auto& e : edges) {
    weights.at(e.first) += intersections(e.first, e.second).size();
    weights.at(e.second) += intersections(e.first, e.second).size();
  }
  return weights;
}

int PickCliqueOrderHelper(const std::vector<std::vector<int>>& cliques_sorted,
                          int root_in,
                          SymmetricMatrix<vector<int>>* intersections_ptr,
                          vector<vector<int>>* separators,
                          std::vector<int>* order, RootedTree* tree_ptr) {
  auto& tree = *tree_ptr;
  auto& intersections = *intersections_ptr;
  size_t n = cliques_sorted.size();
  assert(root_in < static_cast<int>(n));

  vector<int> visited(n);
  for (auto& b : visited) {
    b = 0;
  }

  std::stack<size_t> node_stack;
  int root = root_in;
  if (root < 0) {
    root = 0;
  }
  node_stack.push(root);

  using Path = vector<Edge>;
  vector<Path> paths;
  vector<Edge> edges;

  order->clear();

  while (order->size() < n) {
    size_t active = node_stack.top();

    if (visited.at(active) == 0) {
      order->push_back(active);
      visited.at(active) = 1;
      tree.parent.at(active) = active;
      tree.height.at(active) = 0;
    }

    // Find unvisited neighbor with maximum weight.
    size_t max_weight = 1;
    vector<int> argmax;
    for (size_t i = 0; i < cliques_sorted.size(); i++) {
      if (i == active) {
        continue;
      }

      // Weight is the size of intersection.
      if (intersections(active, i).size() == 0) {
        IntersectionOfSorted(cliques_sorted.at(active), cliques_sorted.at(i),
                             &intersections(active, i));
      }

      if (intersections(active, i).size() >= max_weight && !visited.at(i)) {
        if (intersections(active, i).size() > max_weight) {
          argmax.clear();
          max_weight = intersections(active, i).size();
        }
        argmax.push_back(i);
      }
    }

    for (auto e : argmax) {
      separators->at(e) = intersections(active, e);
      node_stack.push(e);
      order->push_back(e);
      visited.at(e) = 1;
      edges.emplace_back(active, e);
      tree.parent.at(e) = active;
      tree.height.at(e) = tree.height.at(active) + 1;
    }

    if (argmax.size() == 0) {
      node_stack.pop();
      if (node_stack.size() == 0) {
        auto node = GetUnvisited(visited);
        if (node == -1) {
          break;
        } else {
          node_stack.push(node);
        }
      }
    }
  }

  auto weights = GetMaxWeightedDegreeNode(n, edges, intersections);
  int root_node = std::distance(
      weights.begin(),
      std::max_element(weights.begin(), weights.begin() + weights.size()));

  std::reverse(order->begin(), order->end());
  return root_node;
}

}  // namespace

void FillIn(const RootedTree& tree, int num_variables,
            const std::vector<int>& order, vector<std::vector<int>>* supernodes,
            vector<std::vector<int>>* separators) {
  std::vector<int> eliminated(num_variables);
  int num_cliques = order.size();
  for (auto& e : eliminated) {
    e = num_cliques + 1;
  }

  // Detect if variable is a supernode of clique i and
  // clique j.  If so, apply running intersection property
  // to the path from clique i and to clique j:
  //  1) Make a supernode of the clique closest to the root.
  //  2) Make a separator of all other cliques.
  //
  for (size_t i = 0; i < order.size(); i++) {
    for (int v : supernodes->at(order.at(i))) {
      if (eliminated.at(v) < num_cliques) {
        auto fill_in =
            PathInTree(order.at(i), eliminated.at(v), tree.parent, tree.height);
        for (size_t j = 0; j < fill_in.size() - 1; j++) {
          auto e = fill_in.at(j);
          separators->at(e) = UnionOfSorted(separators->at(e), {v});
        }
        eliminated.at(v) = fill_in.back();
      } else {
        eliminated.at(v) = order.at(i);
      }
    }
  }

  supernodes->clear();
  supernodes->resize(num_cliques);
  for (size_t i = 0; i < eliminated.size(); i++) {
    // TODO(FrankPermenter): Remove this check if we refactor
    // to require that require variable set = [0, ..., GetMax(Vars)].
    // As is, variables are only a subset and hence the "eliminated"
    // vector has spurious entries.
    if (eliminated.at(i) < num_cliques) {
      supernodes->at(eliminated.at(i)).push_back(i);
    }
  }
  Sort(separators);
  Sort(supernodes);
}

template <typename T>
auto FindSupernode(const std::vector<int>& separator, const T& b, const T& c,
                   vector<int>* intersection) {
  if (separator.size() == 0) {
    return c;
  }
  // TODO(FrankPermenter): Process separators in elimination order
  // so we do not search over all supernodes.
  for (auto i = b; i != c; ++i) {
    IntersectionOfSorted(separator, *i, intersection);
    if (intersection->size() == separator.size()) {
      return i;
    }
  }
  return c;
}

void GetCliqueEliminationOrder(const vector<vector<int>>& cliques_sorted,
                               int root, vector<int>* order,
                               vector<vector<int>>* supernodes,
                               vector<vector<int>>* separators,
                               RootedTree* tree) {
  size_t n = cliques_sorted.size();
  order->clear();
  order->resize(n);
  separators->clear();
  separators->resize(n);
  SymmetricMatrix<vector<int>> intersections(n);
  int better_root = PickCliqueOrderHelper(cliques_sorted, root, &intersections,
                                          separators, order, tree);

  if (root == -1) {
    order->clear();
    order->resize(n);
    separators->clear();
    separators->resize(n);
    RootedTree tree_i(n);
    PickCliqueOrderHelper(cliques_sorted, better_root, &intersections,
                          separators, order, &tree_i);
    *tree = tree_i;
  }

  supernodes->resize(n);
  for (auto& e : *order) {
    supernodes->at(e).resize(cliques_sorted.at(e).size() -
                             separators->at(e).size());
    if (supernodes->at(e).size() > 0) {
      std::set_difference(cliques_sorted.at(e).begin(),
                          cliques_sorted.at(e).end(), separators->at(e).begin(),
                          separators->at(e).end(), supernodes->at(e).begin());
    }
  }
}

void PickCliqueOrder(const vector<vector<int>>& cliques_sorted, int root,
                     vector<int>* order, vector<vector<int>>* supernodes,
                     vector<vector<int>>* separators,
                     vector<vector<vector<int>>>* post_order_pointer) {
  size_t n = cliques_sorted.size();
  RootedTree tree(n);
  GetCliqueEliminationOrder(cliques_sorted, root, order, supernodes, separators,
                            &tree);
  int num_vars = GetMax(cliques_sorted) + 1;
  FillIn(tree, num_vars, *order, supernodes, separators);

  if (post_order_pointer) {
    int count = 0;
    auto& post_order = *post_order_pointer;
    for (auto& e : *separators) {
      std::vector<int> intersection;
      auto end = supernodes->end();
      auto ptr = FindSupernode(e, supernodes->begin(), end, &intersection);
      if (ptr != end) {
        int match_index = std::distance(supernodes->begin(), ptr);
        post_order.at(match_index).push_back(intersection);
      }
      count++;
    }
  }
}

}  // namespace conex
