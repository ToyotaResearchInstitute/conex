#include "conex/triangular_matrix_workspace.h"

namespace conex {

using T = TriangularMatrixWorkspace;

namespace {
template <typename T>
int LookupSuperNode(const T& o, int index, int start) {
  return o.variable_to_supernode_[index];
}

template <typename T>
double* LookupAddress(T& o, int r, int c) {
  int node = LookupSuperNode(o, c, 0);
  int node_r = LookupSuperNode(o, r, 0);

  int j = o.variable_to_supernode_position_[c];
  if (node == node_r) {
    int i = o.variable_to_supernode_position_[r];
    return &o.diagonal[node](i, j);
  }

  int cnt = 0;
  for (auto si : o.separators[node]) {
    if (si == r) {
      return &o.off_diagonal[node](j, cnt);
    }
    cnt++;
  }
  throw std::runtime_error(
      "Specified entry of sparse matrix is not accessible.");
}

}  // namespace

TriangularMatrixWorkspace::TriangularMatrixWorkspace(
    const std::vector<Clique>& path_, const std::vector<int>& supernode_size_)
    : supernode_size(supernode_size_) {
  N = std::accumulate(supernode_size.begin(), supernode_size.end(), 0);
  variable_to_supernode_.resize(N);
  variable_to_supernode_position_.resize(N);

  // separators.resize(path_.size());
  // for (auto& si : separators) {
  //  for (size_t i = supernode_size.at(cnt); i < path_.at(cnt).size(); i++) {
  //    si.push_back(path_.at(cnt).at(i));
  //  }
  //  cnt++;
  //}

  int cnt = 0;
  int var = 0;
  snodes.resize(path_.size());
  for (auto& si : snodes) {
    si.resize(supernode_size.at(cnt));
    for (int i = 0; i < supernode_size.at(cnt); i++) {
      if (var >= N) {
        std::runtime_error("Invalid variable index.");
      }
      si.at(i) = path_.at(cnt).at(i);
      variable_to_supernode_[var] = cnt;
      variable_to_supernode_position_[var] = i;
      var++;
    }
    cnt++;
  }

  separators.resize(path_.size());
  column_intersections.resize(snodes.size() - 1);
  intersection_position.resize(snodes.size() - 1);
  cnt = 0;

  // For each supernode [sn], find cliques that overlap. Store
  // this using two list of lists:.
  //  seperator_list(supernode) = list of separators
  //  column_intersection(supernode) = list of (i, j) pairs indexed by k, where
  //  supernode[i] = separator_list(supernode)[k][j]
  for (auto& sep_i : separators) {
    int seperator_size = path_.at(cnt).size() - supernode_size.at(cnt);
    sep_i.resize(seperator_size);
    for (int i = 0; i < seperator_size; i++) {
      int var = path_.at(cnt).at(i + supernode_size.at(cnt));
      sep_i[i] = var;

      int sn = variable_to_supernode_[var] - 1;
#ifndef NDEBUG
      if (sn < 0) {
        throw std::runtime_error("Invalid supernode index.");
      }
#endif

#ifndef NDEBUG
      if (cnt > sn) {
        throw std::runtime_error(
            "Supernode has already been eliminated. The sparsity pattern is "
            "malformed.");
      }
#endif

      // Create list for this separator if supernode doesn't have one.
      if (column_intersections[sn].size() == 0 ||
          column_intersections[sn].back() != cnt) {
        column_intersections[sn].push_back(cnt);
        intersection_position[sn].emplace_back(
            std::vector<std::pair<int, int>>());
      }
      std::pair<int, int> pair{variable_to_supernode_position_[var], i};
      intersection_position[sn].back().push_back(pair);
    }
    cnt++;
  }

  // TODO(FrankPermenter): Remove this.
  for (auto& l : column_intersections) {
    std::reverse(l.begin(), l.end());
  }
  for (auto& l : intersection_position) {
    std::reverse(l.begin(), l.end());
  }
}

void Initialize(TriangularMatrixWorkspace* o, double* data_start) {
  double* data = data_start;
  for (size_t j = 0; j < o->snodes.size(); j++) {
    o->diagonal.emplace_back(data, o->supernode_size.at(j),
                             o->supernode_size.at(j));

    data += o->SizeOfSupernode(j);
    o->off_diagonal.emplace_back(data, o->supernode_size.at(j),
                                 o->separators.at(j).size());
    data += o->SizeOfSeparator(j);
  }

  o->seperator_diagonal.resize(o->snodes.size());
  for (size_t j = 0; j < o->snodes.size(); j++) {
    o->S_S(j, &o->seperator_diagonal.at(j));
  }

  // Use reserve so that we can call default constructor of LLT objects.
  o->llts.reserve(o->snodes.size());

  o->temporaries.resize(o->snodes.size());
  for (size_t j = 0; j < o->snodes.size(); j++) {
    o->temporaries.at(j).resize(o->separators.at(j).size());
  }
}

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

}  // namespace conex
