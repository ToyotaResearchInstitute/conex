#include "conex/supernodal_solver.h"
#include "conex/clique_ordering.h"

#include <iostream>
#include <map>

#include <Eigen/Dense>

namespace conex {

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Permutation = Eigen::PermutationMatrix<-1>;
using T = TriangularMatrixOperations;
using std::vector;

namespace {

vector<int> Relabel(const vector<int>& x, const vector<int>& labels) {
  vector<int> y;
  for (auto& xi : x) {
    y.push_back(labels.at(xi));
  }
  return y;
}

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

class TriangularMatrixColumnOperations {
 public:
  TriangularMatrixColumnOperations(T::Matrix* mat) : mat_(mat) {
    dense_size = mat_->supernode_size.at(0);
  }
  void NextColumn() {
    supernode_column_++;
    column++;
    dense_size--;
    if (supernode_column_ >= mat_->supernode_size.at(supernode_index_)) {
      supernode_index_++;
      supernode_column_ = 0;
      dense_size = mat_->supernode_size.at(supernode_index_);
    }
  }

  void Rescale(double scale) {
    mat_->supernodes.at(supernode_index_).col(supernode_column_).array() *=
        scale;
    mat_->separator.at(supernode_index_).row(supernode_column_).array() *=
        scale;
  }

  double Diagonal() {
    return mat_->supernodes.at(supernode_index_)(supernode_column_,
                                                 supernode_column_);
  }

  // Applies the operation b = b - L.col(i) * y (i)
  void SubtractWeightedColumn(VectorXd* b, double y) {
    b->segment(column, dense_size) -= y * mat_->supernodes.at(supernode_index_)
                                              .col(supernode_column_)
                                              .tail(dense_size);

    int offset = mat_->supernode_size.at(supernode_index_);
    for (int i = 0; i < mat_->separator.at(supernode_index_).cols(); i++) {
      int row = mat_->path.at(supernode_index_).at(offset + i);
      (*b)(row) -=
          y * mat_->separator.at(supernode_index_)(supernode_column_, i);
    }
  }

  std::vector<int> NonzeroRows() {
    std::vector<int> y;
    for (int i = 1; i < dense_size; i++) {
      y.push_back(column + i);
    }
    int offset = mat_->supernode_size.at(supernode_index_);
    for (int i = 0; i < mat_->separator.at(supernode_index_).cols(); i++) {
      int row = mat_->path.at(supernode_index_).at(offset + i);
      y.push_back(row);
    }
    return y;
  }

  // double DotProduct(const Eigen::MatrixXd& x) {
  //  double y = 0;
  //  mat_->supernodes.at(supernode_index_).col(supernode_column_).tail(dense_size).dot(
  //      x.segment(column, dense_size);
  //}

 private:
  int supernode_index_ = 0;
  int dense_size = 0;
  int column = 0;
  int supernode_column_ = 0;
  SparseTriangularMatrix* mat_;
};

template <typename T>
int LookupSuperNode(const T& o, int index, int start) {
  for (int j = static_cast<int>(o.snodes.size()) - 1; j >= 0; j--) {
    if (o.snodes.at(j).size() > 0) {
      if (index >= o.snodes.at(j).at(0)) {
        return j;
      }
    }
  }
  throw "Sparse matrix is malformed: invalid supernode partition.";
}

double Get(const SparseTriangularMatrix& o, int i, int j) {
  if (j > i) {
    return 0;
  }
  // Find the supernode that owns the column.
  int node = LookupSuperNode(o, j, 0);

  // Apply offsets.
  int offset_i = i - o.path.at(node).at(0);
  int offset_j = j - o.path.at(node).at(0);
  if ((offset_i < o.supernode_size.at(node)) &&
      (offset_j < o.supernode_size.at(node))) {
    return o.supernodes.at(node)(offset_i, offset_j);
  }
  for (size_t k = o.supernode_size.at(node); k < o.path.at(node).size(); k++) {
    if (i == o.path.at(node).at(k)) {
      return o.separator.at(node)(offset_j, k - o.supernode_size.at(node));
    }
  }
  return 0;
}

class LowerTriangularSuperNodal {
  // We partition matrix into
  //
  // SN1  R1
  // R1   SN2  R2
  // R1   R2   SN3
  //
  // Where R1 is sparse.
 public:
  LowerTriangularSuperNodal(SparseTriangularMatrix* mat)
      : supernodes_(mat->supernodes),
        supernode_size_(mat->supernode_size),
        path_(mat->path),
        separator_(mat->separator) {
    Init();
  }

  void Init() {
    // for (int j = static_cast<int>(path_.size()) - 1; j >= 0; j--) {
    //   supernodes_.at(j).resize(supernode_size_.at(j), supernode_size_.at(j));
    //   separator_.at(j).resize(supernode_size_.at(j),  path_.at(j).size() -
    //   supernode_size_.at(j));
    // }
  }

  // The smallest super-node less than i.
  int LookupSuperNode(int index, int start) const {
    for (int j = static_cast<int>(path_.size()) - 1; j >= 0; j--) {
      if (index >= path_.at(j).at(0)) {
        return j;
      }
    }
    throw "Sparse matrix is malformed: invalid supernode partition.";
  }

  double Get(int i, int j) const {
    if (j > i) {
      return 0;
    }
    // Find the supernode that owns the column.
    int node = LookupSuperNode(j, 0);

    // Apply offsets.
    int offset_i = i - path_.at(node).at(0);
    int offset_j = j - path_.at(node).at(0);
    if ((offset_i < supernode_size_.at(node)) &&
        (offset_j < supernode_size_.at(node))) {
      return supernodes_.at(node)(offset_i, offset_j);
    }
    for (size_t k = supernode_size_.at(node); k < path_.at(node).size(); k++) {
      if (i == path_.at(node).at(k)) {
        return separator_.at(node)(offset_j, k - supernode_size_.at(node));
      }
    }
    return 0;
  }

  void Increment(double val, int i, int j, int supernode_index = -1) {
    if (val == 0) {
      return;
    }
    assert(j <= i);

    int node = supernode_index;
    // Find the supernode that owns storage for column j.
    if (node == -1) {
      node = LookupSuperNode(j, 0);
    }

    // Apply offsets.
    int offset_i = i - path_.at(node).at(0);
    int offset_j = j - path_.at(node).at(0);

    // i is also in the supernode.
    if ((offset_i < supernode_size_.at(node)) &&
        (offset_j < supernode_size_.at(node))) {
      supernodes_.at(node)(offset_i, offset_j) += val;
      return;
    }

    // i is in a separator.
    for (size_t k = supernode_size_.at(node); k < path_.at(node).size(); k++) {
      if (i == path_.at(node).at(k)) {
        separator_.at(node)(offset_j, k - supernode_size_.at(node)) += val;
        return;
      }
    }

    throw "Specified entry of sparse matrix is not accesible.";
  }

 public:
  using MapT = Eigen::Map<Eigen::MatrixXd>;
  std::vector<MapT>& supernodes_;
  std::vector<int>& supernode_size_;
  std::vector<Clique>& path_;
  std::vector<MapT>& separator_;
};

std::vector<int> ResidualSize(std::vector<Clique>& path) {
  std::vector<int> y;
  for (size_t j = 0; j < path.size() - 1; j++) {
    std::vector<int> temp;
    IntersectionOfSorted(path.at(j), path.at(j + 1), &temp);
    y.push_back(path.at(j).size() - temp.size());
  }
  y.push_back(path.back().size());
  return y;
}

}  // namespace

void IntersectionOfSorted(const std::vector<int>& v1,
                          const std::vector<int>& v2, std::vector<int>* v3) {
  v3->clear();
  std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(),
                        back_inserter(*v3));
}

SparseTriangularMatrix GetFillInPattern(
    int N, const std::vector<Clique>& cliques_input) {
  auto mat = MakeSparseTriangularMatrix(N, cliques_input);

  for (int j = static_cast<int>(mat.path.size()) - 1; j >= 0; j--) {
    // Initialize columns of super nodes.
    mat.supernodes.at(j).setConstant(1);
    mat.separator.at(j).setConstant(1);

    // Update other columns: the (seperator, seperator) components.
    int index = 0;
    auto s_s = mat.workspace_.seperator_diagonal.at(j);
    int n = mat.path.at(j).size();
    for (int i = mat.supernode_size.at(j); i < n; i++) {
      for (int k = i; k < n; k++) {
        *s_s.at(index++) += 1;
      }
    }
  }
  return mat;
}

std::vector<Clique> Permute(std::vector<Clique>& path,
                            std::vector<int>& permutation) {
  auto y = path;
  for (size_t i = 0; i < path.size(); i++) {
    for (size_t j = 0; j < path.at(i).size(); j++) {
      y.at(i).at(j) = permutation.at(path.at(i).at(j));
    }
  }
  return y;
}

void Sort(std::vector<Clique>* path) {
  for (size_t i = 0; i < path->size(); i++) {
    std::sort(path->at(i).begin(), path->at(i).end());
  }
}

// Want A B  C  D  E
//   Add A_i \cap A_k
//    to all A_j for i < j < k.
//
void RunningIntersectionClosure(std::vector<Clique>* path) {
  if (path->size() < 2) {
    return;
  }
  int n = path->size();
  for (int i = 0; i < n - 2; i++) {
    for (int j = n - 1; j > i + 1; j--) {
      std::vector<int> temp;
      IntersectionOfSorted(path->at(i), path->at(j), &temp);
      if (temp.size() == 0) {
        continue;
      }
      for (int k = j - 1; k > i; k--) {
        path->at(k) = UnionOfSorted(path->at(k), temp);
      }
    }
  }
}

Eigen::MatrixXd TriangularMatrixOperations::ToDense(
    const SparseTriangularMatrix& mat) {
  MatrixXd y(mat.N, mat.N);
  for (int i = 0; i < mat.N; i++) {
    for (int j = 0; j < mat.N; j++) {
      y(i, j) = Get(mat, i, j);
    }
  }
  return y;
}

void TriangularMatrixOperations::SetConstant(SparseTriangularMatrix* mat,
                                             double val) {
  for (auto& n : mat->supernodes) {
    n.array() = val;
  }
  for (auto& n : mat->separator) {
    n.array() = val;
  }
}

SparseTriangularMatrix MakeSparseTriangularMatrix(
    int N, const std::vector<Clique>& path_) {
  auto path = path_;
  Sort(&path);
  RunningIntersectionClosure(&path);
  auto supernode_size = ResidualSize(path);
  return SparseTriangularMatrix(N, path, supernode_size);
}

void T::CholeskyInPlace(SparseTriangularMatrix* C) {
  TriangularMatrixColumnOperations col(C);
  LowerTriangularSuperNodal mat(C);
  double sqrt_d = std::sqrt(col.Diagonal());
  col.Rescale(1.0 / sqrt_d);

  // i: a supernode
  for (int i = 0; i < C->N - 1; i++) {
    // Substract col(2:n) c(2:n)^T
    auto indices = col.NonzeroRows();
    for (size_t k = 0; k < indices.size(); k++) {
      double weight = -mat.Get(indices.at(k), i);
      // Decrement column k
      for (size_t j = k; j < indices.size(); j++) {
        mat.Increment(weight * mat.Get(indices.at(j), i), indices.at(j),
                      indices.at(k));
      }
    }
    col.NextColumn();
    double sqrt_d = std::sqrt(col.Diagonal());
    col.Rescale(1.0 / sqrt_d);
  }
}

// Apply L  inverse.
// 1
// 1  1
// 1  1  1

// L
// B in
VectorXd T::ApplyInverseOfTranspose(SparseTriangularMatrix* mat,
                                    const VectorXd& b) {
  assert(b.rows() == mat->N);
  int n = b.rows();
  VectorXd y(n);
  auto res = b;
  LowerTriangularSuperNodal L(mat);

  y(n - 1) = res(n - 1) / L.Get(n - 1, n - 1);
  for (int i = n - 2; i >= 0; i--) {
    for (int j = 0; j < i + 1; j++) {
      res(j) = res(j) - L.Get(i + 1, j) * y(i + 1);
    }
    y(i) = res(i) / L.Get(i, i);
  }
  return y;
}

VectorXd T::ApplyInverse(SparseTriangularMatrix* mat, const VectorXd& b) {
  assert(b.rows() == mat->N);
  int n = b.rows();
  VectorXd y(n);
  auto res = b;
  LowerTriangularSuperNodal L(mat);

  TriangularMatrixColumnOperations col(mat);
  y(0) = res(0) / col.Diagonal();
  for (int i = 1; i < n; i++) {
    // Iterate over non-zero entries of column.
    // for (int j = i; j < n; j++) {
    //   res(j) = res(j) - L.Get(j, i - 1) * y(i - 1);
    // }
    col.SubtractWeightedColumn(&res, y(i - 1));

    col.NextColumn();
    y(i) = res(i) / col.Diagonal();
  }
  return y;
}

std::vector<int> UnionOfSorted(const std::vector<int>& x1,
                               const std::vector<int>& x2) {
  std::vector<int> y;
  set_union(x1.begin(), x1.end(), x2.begin(), x2.end(), inserter(y, y.end()));
  return y;
}

MatrixData GetData(const vector<vector<int>>& cliques) {
  vector<vector<int>> separators;
  vector<vector<int>> supernodes;
  MatrixData d;
  d.cliques = cliques;
  Sort(&d.cliques);
  auto& order = d.clique_order;

  PickCliqueOrder(d.cliques, 0, &order, &supernodes, &separators);

  d.permutation.resize(GetMax(cliques) + 1);
  d.permutation_inverse.resize(GetMax(cliques) + 1);
  int i = 0;
  for (auto& e : order) {
    for (auto& sn_ii : supernodes.at(e)) {
      d.permutation_inverse.at(i) = sn_ii;
      d.permutation.at(sn_ii) = i;
      i++;
    }
  }

  auto& separators_ = d.separators;
  auto& supernodes_ = d.supernodes;

  supernodes_.resize(order.size());
  separators_.resize(order.size());

  i = 0;

  for (auto e : order) {
    supernodes_.at(i) = Relabel(supernodes.at(e), d.permutation);
    separators_.at(i) = Relabel(separators.at(e), d.permutation);
    i++;
  }

  Sort(&separators_);
  Sort(&supernodes_);

  int cnt = 0;
  auto& supernode_size = d.supernode_size;
  supernode_size.resize(order.size());
  for (auto& si : supernode_size) {
    si = supernodes_.at(cnt).size();
    cnt++;
  }
  d.N = std::accumulate(supernode_size.begin(), supernode_size.end(), 0);
  for (size_t i = 0; i < d.cliques.size(); i++) {
    d.cliques.at(i) = UnionOfSorted(supernodes_.at(i), separators_.at(i));
  }
  return d;
}

}  // namespace conex
