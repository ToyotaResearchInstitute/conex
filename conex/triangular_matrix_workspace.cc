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

  separators.resize(path_.size());
  int cnt = 0;
  for (auto& si : separators) {
    for (size_t i = supernode_size.at(cnt); i < path_.at(cnt).size(); i++) {
      si.push_back(path_.at(cnt).at(i));
    }
    cnt++;
  }

  cnt = 0;
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

  std::vector<std::vector<int>> separators2;
  separators2.resize(path_.size());
  cnt = 0;
  column_intersections.resize(snodes.size() - 1);
  intersection_position.resize(snodes.size() - 1);
  cnt = separators2.size() - 1;
  cnt = 0; 

  for (auto& si : separators2) {
     for (size_t ii = supernode_size.at(cnt); ii < path_.at(cnt).size(); ii++) {
        int i = ii; //path_.at(cnt).size() - 1 - ii;
        if (i < 0) {
          throw std::runtime_error("NOP!");
         }
       int var = path_.at(cnt).at(i);
       si.push_back(var);

       // Need to decide if supernode already has a list for
       // Fill a list of list 
       //  list(supernode) = list of pairs 
       //  list(supernode) = list of separators2
       int sn = variable_to_supernode_.at(var) - 1;
       if (sn < 0) {
        continue;
       }
       if (column_intersections.at(sn).size() == 0 ||  
           column_intersections.at(sn).back() != cnt) {
         column_intersections.at(sn).push_back(cnt);
         intersection_position.at(sn).emplace_back(std::vector<std::pair<int, int>>());
       }
       std::pair<int, int> pair{variable_to_supernode_position_[var], si.size() - 1};
       intersection_position.at(sn).back().push_back(pair);
     }
    cnt++;
  }
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
  o->SetIntersections();

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
  return;
  DUMP("BEFORE");
  DUMP(column_intersections);
  DUMP(intersection_position);

  auto i1 = column_intersections;
  auto i2 = intersection_position;

  column_intersections.clear();
  intersection_position.clear();
  column_intersections.resize(snodes.size() - 1);
  intersection_position.resize(snodes.size() - 1);
  for (int i = static_cast<int>(diagonal.size() - 2); i >= 0; i--) {
    for (int j = i; j >= 0; j--) {
      const auto& temp = IntersectionOfSupernodeAndSeparator(i + 1, j);
      if (temp.size() > 0) {
        // For each supernode i, save separator j.
        // For each supernode i, save (supernode position, separator position)
        column_intersections.at(i).push_back(j);
        intersection_position.at(i).push_back(temp);
      }
    }
  }
  DUMP("AFTER");
  DUMP(column_intersections);
  DUMP(intersection_position);
  column_intersections = i1;
  intersection_position = i2; 
}
}  // namespace conex
