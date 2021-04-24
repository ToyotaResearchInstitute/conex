#include "conex/triangular_matrix_workspace.h"

namespace conex {

using T = TriangularMatrixWorkspace;

namespace {
template <typename T>
int LookupSuperNode(const T& o, int index, int start) {
  return o.variable_to_supernode_.at(index);
}

template <typename T>
double* LookupAddress(T& o, int r, int c) {
  int node = LookupSuperNode(o, c, 0);
  int node_r = LookupSuperNode(o, r, 0);

  int j = o.variable_to_supernode_position_.at(c);
  if (node == node_r) {
    int i = o.variable_to_supernode_position_.at(r);
    return &o.diagonal.at(node)(i, j);
  }

  int cnt = 0;
  for (auto si : o.separators.at(node)) {
    if (si == r) {
      return &o.off_diagonal.at(node)(j, cnt);
    }
    cnt++;
  }
  throw std::runtime_error(
      "Specified entry of sparse matrix is not accessible.");
}

}  // namespace

void TriangularMatrixWorkspace::S_S(int clique, std::vector<double*>* y) {
  auto& s = separators.at(clique);
  int size = .5 * (s.size() * s.size() + s.size());
  y->resize(size);
  int cnt = 0;
  for (size_t j = 0; j < s.size(); j++) {
    for (size_t i = j; i < s.size(); i++) {
      (*y)[cnt++] = LookupAddress(*this, s[i], s[j]);
    }
  }
}

// Given supernode N and separator S, returns (i, j) pairs for which
// S(i) == N(j).
using Match = std::pair<int, int>;
std::vector<Match> T::IntersectionOfSupernodeAndSeparator(int supernode,
                                                          int seperator) const {
  std::vector<Match> y;
  int i = 0;
  for (auto si : snodes.at(supernode)) {
    int j = 0;
    for (auto sj : separators.at(seperator)) {
      if (si == sj) {
        y.emplace_back(i, j);
      }
      j++;
    }
    i++;
  }
  return y;
}

void T::SetIntersections() {
  column_intersections.resize(snodes.size() - 1);
  intersection_position.resize(snodes.size() - 1);
  for (int i = static_cast<int>(diagonal.size() - 2); i >= 0; i--) {
    for (int j = i; j >= 0; j--) {
      const auto& temp = IntersectionOfSupernodeAndSeparator(i + 1, j);
      if (temp.size() > 0) {
        column_intersections.at(i).push_back(j);
        intersection_position.at(i).push_back(temp);
      }
    }
  }
}
}  // namespace conex
