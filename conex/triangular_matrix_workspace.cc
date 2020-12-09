#include "conex/triangular_matrix_workspace.h"

namespace conex {


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

} // namespace conex
