#include "conex/tree_gram.h"

#include <iostream>
#include <map>

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXi;
using Permutation = Eigen::PermutationMatrix<-1>;
using T = TriangularMatrixOperations;

namespace {




class TriangularMatrixColumnOperations {
 public:
  TriangularMatrixColumnOperations(T::Matrix* mat) : mat_(mat) {
    dense_size = mat_->residual_size.at(0);
  }
  void NextColumn() {
    supernode_column_++;
    column++;
    dense_size--;
    if (supernode_column_ >= mat_->residual_size.at(supernode_index_)) {
      supernode_index_++;
      supernode_column_ = 0;
      dense_size = mat_->residual_size.at(supernode_index_);
    }
  }

  void Rescale(double scale) {
    mat_->supernodes.at(supernode_index_).col(supernode_column_).array() *= scale;
    mat_->separator.at(supernode_index_).row(supernode_column_).array() *= scale;
  }

  double Diagonal() {
    return mat_->supernodes.at(supernode_index_)(supernode_column_, supernode_column_);
  }

  // Applies the operation b = b - L.col(i) * y (i)
  void SubtractWeightedColumn(VectorXd* b, double y) {
    b->segment(column, dense_size) -= 
        y * mat_->supernodes.at(supernode_index_).col(supernode_column_).tail(dense_size);

    int offset = mat_->residual_size.at(supernode_index_);
    for (int i = 0; i < mat_->separator.at(supernode_index_).cols(); i++) { 
      int row = mat_->path.at(supernode_index_).at(offset+i);
      (*b)(row) -= y * mat_->separator.at(supernode_index_)(supernode_column_, i);
    }
  }

  std::vector<int> NonzeroRows() {
    std::vector<int> y;
    for (int i = 1; i < dense_size; i++) { 
      y.push_back(column + i);
    }
    int offset = mat_->residual_size.at(supernode_index_);
    for (int i = 0; i < mat_->separator.at(supernode_index_).cols(); i++) { 
      int row = mat_->path.at(supernode_index_).at(offset+i);
      y.push_back(row);
    }
    return y;
  }


  //double DotProduct(const Eigen::MatrixXd& x) {
  //  double y = 0;
  //  mat_->supernodes.at(supernode_index_).col(supernode_column_).tail(dense_size).dot(
  //      x.segment(column, dense_size);
  //}

 private:
  int supernode_index_ = 0;
  int dense_size = 0;
  int column  = 0;
  int supernode_column_ = 0;
  SparseTriangularMatrix* mat_;
};


class LowerTriangularSuperNodal {
  // We partition matrix into
  //
  // SN1  R1
  // R1   SN2  R2
  // R1   R2   SN3
  //
  // Where R1 is sparse.
 public:
  LowerTriangularSuperNodal(SparseTriangularMatrix* mat) :
  supernodes_(mat->supernodes),
  residual_size_(mat->residual_size),
  path_(mat->path),
  separator_(mat->separator) {
    Init();
  }

  void Init() {
    supernodes_.resize(path_.size());
    separator_.resize(path_.size());
    for (int j = static_cast<int>(path_.size()) - 1; j >= 0; j--) {
      supernodes_.at(j).resize(residual_size_.at(j), residual_size_.at(j));
      separator_.at(j).resize(residual_size_.at(j),  path_.at(j).size() - residual_size_.at(j));
    }
  }

  // The smallest super-node less than i.
  int LookupSuperNode(int index, int start) const {
    for (int j = static_cast<int>(path_.size()) - 1; j >= 0; j--) {
      if (index >= path_.at(j).at(0)) {
        return j;
      }
    }
    assert(0);
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
    if ((offset_i < residual_size_.at(node))  &&
        (offset_j < residual_size_.at(node))) {
      return supernodes_.at(node)(offset_i, offset_j);
    }
    for (size_t k = residual_size_.at(node); k < path_.at(node).size(); k++) {
      if (i == path_.at(node).at(k)) {
        return separator_.at(node)(offset_j,  k - residual_size_.at(node));
      }
    }
    return 0;
  }

  void Increment(double val, int i, int j) {
    if (val == 0) {
      return;
    }
    // Find the supernode that owns the column.
    int node = LookupSuperNode(j, 0);

    // Apply offsets.
    int offset_i = i - path_.at(node).at(0);
    int offset_j = j - path_.at(node).at(0);
    if ((offset_i < residual_size_.at(node))  &&
        (offset_j < residual_size_.at(node))) {
      supernodes_.at(node)(offset_i, offset_j) += val;
    }
    for (size_t k = residual_size_.at(node); k < path_.at(node).size(); k++) {
      if (i == path_.at(node).at(k)) {
        separator_.at(node)(offset_j, k - residual_size_.at(node)) += val;
      }
    }
    bool entry_not_in_sparsity_pattern = false;
    assert(!entry_not_in_sparsity_pattern);
  }

  // Traverse path backwards.
  void Assemble() {
    for (int j = static_cast<int>(path_.size()) - 1; j >= 0; j--) {
      // Initialize columns of super nodes.
      auto p_i = path_.at(j);
      supernodes_.at(j).setConstant(1);
      separator_.at(j).setConstant(1);

      // Update other columns.
      for (size_t i = residual_size_.at(j); i < p_i.size(); i++) {
        for (size_t k = i; k < p_i.size(); k++) {
          // TODO(FrankPermenter): use an iterator instead of random access.
          Increment(1, p_i.at(k), p_i.at(i));
        }
      }
    }
  }

 public:
  std::vector<MatrixXd>& supernodes_;
  std::vector<int>& residual_size_;
  std::vector<Clique>& path_;
  std::vector<MatrixXd>& separator_;
};


std::vector<int> ResidualSize(std::vector<Clique>& path) {
  std::vector<int> y;
  for (size_t j = 0; j < path.size() - 1; j++) {
    std::vector<int> temp;
    IntersectionOfSorted(path.at(j), path.at(j+1), &temp);
    y.push_back(path.at(j).size()  - temp.size());
  }
  y.push_back(path.back().size());
  return y;
}

} // namespace


void IntersectionOfSorted(const std::vector<int>& v1,
                  const std::vector<int>& v2,
                  std::vector<int>* v3){
    v3->clear();
    std::set_intersection(v1.begin(), v1.end(),
                          v2.begin(), v2.end(),
                          back_inserter(*v3));
}


SparseTriangularMatrix GetFillInPattern(int N, const std::vector<Clique>& cliques_input) {
  auto mat = MakeSparseTriangularMatrix(N, cliques_input);
  LowerTriangularSuperNodal node(&mat);
  node.Assemble();
  return mat;
}

std::vector<Clique> Permute(std::vector<Clique>& path, std::vector<int>& permutation) {
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


Eigen::MatrixXd
TriangularMatrixOperations::Multiply(SparseTriangularMatrix& mat_in,
                                     const Eigen::MatrixXd& x) {
  auto mat = mat_in;
  assert(mat.N = x.rows());
  LowerTriangularSuperNodal node(&mat);
  MatrixXd y(mat.N, x.cols());
  for (int i = 0; i < mat.N; i++) {
    y(i, 0) = node.Get(i, 0) * x(0, 0);
    for (int j = 1; j < mat.N; j++) {
      y(i, 0) = node.Get(i, j) * x(j, 0);
    }
  }
  return y;
}

Eigen::MatrixXd TriangularMatrixOperations::ToDense(const SparseTriangularMatrix& mat_in) {
  auto mat = mat_in;
  MatrixXd y(mat.N, mat.N);
  LowerTriangularSuperNodal node(&mat);
  for (int i = 0; i < mat.N; i++) {
    for (int j = 0; j < mat.N; j++) {
      y(i, j) = node.Get(i, j);
    }
  }
  return y;
}

void TriangularMatrixOperations::SetConstant(SparseTriangularMatrix* mat, double val) {
  for (auto& n : mat->supernodes) {
    n.array() = val;
  }
  for (auto& n : mat->separator) {
    n.array() = val;
  }
}

SparseTriangularMatrix MakeSparseTriangularMatrix(int N, const std::vector<Clique>& path) {
  SparseTriangularMatrix mat;
  mat.path = path;
  mat.N = N;

  Sort(&mat.path);
  RunningIntersectionClosure(&mat.path);
  mat.residual_size = ResidualSize(mat.path);
  auto& residuals = mat.residual_size;
  int cnt = 0;
  mat.permutation.resize(N);
  for (size_t k = 0; k < residuals.size(); k++) {
    for (int i = 0; i < residuals.at(k); i++) {
      mat.permutation.at(mat.path.at(k).at(i)) = cnt++;
    }
  }

  mat.supernodes.resize(mat.path.size());
  mat.separator.resize(mat.path.size());
  for (int j = static_cast<int>(mat.path.size()) - 1; j >= 0; j--) {
    mat.supernodes.at(j).resize(mat.residual_size.at(j), mat.residual_size.at(j));
    mat.separator.at(j).resize(mat.residual_size.at(j), mat.path.at(j).size() - mat.residual_size.at(j));
  }

  return mat;
}


void T::CholeskyInPlace(SparseTriangularMatrix* C) {
  TriangularMatrixColumnOperations col(C);
  LowerTriangularSuperNodal mat(C);
  double sqrt_d = std::sqrt(col.Diagonal());
  col.Rescale(1.0/sqrt_d);
  
  // i: a supernode
  for (int i = 0; i < C->N - 1; i++) {
    // Substract col(2:n) c(2:n)^T
    auto indices = col.NonzeroRows();
    for (size_t k = 0; k < indices.size(); k++) {
      double weight = -mat.Get(indices.at(k), i);
      // Decrement column k
      for (size_t j = k; j < indices.size(); j++) {
        mat.Increment(weight * mat.Get(indices.at(j), i), 
                                       indices.at(j), indices.at(k));
      }
    }
    col.NextColumn();
    double sqrt_d = std::sqrt(col.Diagonal());
    col.Rescale(1.0/sqrt_d);
  }
}




// Apply L  inverse.
// 1
// 1  1
// 1  1  1

// L
// B in
VectorXd T::ApplyInverseOfTranspose(SparseTriangularMatrix* mat, const VectorXd& b) {
  assert(b.rows() == mat->N);
  int n = b.rows();
  VectorXd y(n);
  auto res = b;
  LowerTriangularSuperNodal L(mat);

  y(n-1) = res(n-1) / L.Get(n-1, n-1);
  for (int i = n-2; i >= 0; i--) {
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


std::vector<int> UnionOfSorted(const std::vector<int>& x1, const std::vector<int>& x2) {
  std::vector<int> y;
  set_union(x1.begin(), x1.end(), x2.begin(), x2.end(), inserter(y, y.end()));
  return y;
}
