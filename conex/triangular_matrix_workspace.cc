#include "conex/triangular_matrix_workspace.h"

namespace conex {

using T = TriangularMatrixWorkspace;

namespace {
template <typename T>
int LookupSuperNode(const T& o, int index, int start) {
  for (int j = static_cast<int>(o.snodes.size()) - 1; j >= 0; j--) {
    if (o.snodes.at(j).size() > 0) {
      if (index >= o.snodes.at(j).at(0)) {
        return j;
      }
    }
  }
  assert(0);
}

template <typename T>
double* LookupAddress(T& o, int r, int c) {
  int node = LookupSuperNode(o, c, 0);
  int j = 0;
  for (auto sj : o.snodes.at(node)) {
    if (sj == c) {
      break;
    }
    j++;
  }

  int n = o.diagonal.at(node).rows();
  for (int i = j; i < n; i++) {
    if (o.snodes.at(node).at(i) == r) {
      return &o.diagonal.at(node)(i, j);
    }
  }

  int cnt = 0;
  for (auto si : o.separators.at(node)) {
    if (si == r) {
      return &o.off_diagonal.at(node)(j, cnt);
    }
    cnt++;
  }
  assert(0);
}

}  // namespace

std::vector<double*> TriangularMatrixWorkspace::S_S(int clique) {
  std::vector<double*> y;
  auto& s = separators.at(clique);
  for (size_t j = 0; j < s.size(); j++) {
    for (size_t i = j; i < s.size(); i++) {
      auto temp = LookupAddress(*this, s.at(i), s.at(j));
      y.push_back(temp);
    }
  }
  return y;
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
