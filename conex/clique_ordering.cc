#include "conex/clique_ordering.h"

#include <algorithm>
#include <stack>

#include "conex/debug_macros.h"
#include "conex/supernodal_solver.h"

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
                          std::vector<int>* order) {
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

void PickCliqueOrder(const std::vector<std::vector<int>>& cliques_sorted,
                     int root, std::vector<int>* order,
                     std::vector<std::vector<int>>* supernodes,
                     std::vector<std::vector<int>>* separators) {
  size_t n = cliques_sorted.size();
  order->clear();
  order->resize(n);
  separators->clear();
  separators->resize(n);

  SymmetricMatrix<vector<int>> intersections(n);
  int better_root = PickCliqueOrderHelper(cliques_sorted, root, &intersections,
                                          separators, order);
  if (root == -1) {
    order->clear();
    order->resize(n);
    separators->clear();
    separators->resize(n);
    int r2 = PickCliqueOrderHelper(cliques_sorted, better_root, &intersections,
                                   separators, order);
    assert(r2 == better_root);
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

  vector<size_t> eliminated(GetMax(cliques_sorted) + 1);
  for (auto& e : eliminated) {
    e = n + 1;
  }

  // Fill in
  for (size_t i = 0; i < order->size(); i++) {
    for (auto v : supernodes->at(order->at(i))) {
      if (eliminated.at(v) < n) {
        for (size_t j = eliminated.at(v); j < i; j++) {
          separators->at(order->at(j)).push_back(v);
        }
        auto& sn = supernodes->at(order->at(eliminated.at(v)));
        for (size_t s = 0; s < sn.size(); s++) {
          sn.erase(sn.begin() + s);
          break;
        }
      }
      eliminated.at(v) = i;
    }
  }
}
