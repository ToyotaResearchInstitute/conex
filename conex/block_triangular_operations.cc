#include "block_triangular_operations.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;

using T = BlockTriangularOperations;
namespace {

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
  separator_(mat->separator) { }


  // The smallest super-node less than i.
  int LookupSuperNode(int index, int start) const {
    for (int j = static_cast<int>(path_.size()) - 1; j >= 0; j--) {
      if (index >= path_.at(j).at(0)) {
        return j;
      }
    }
    assert(0);
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
        separator_.at(node)(offset_j,  k - residual_size_.at(node)) += val;
      }
    }
    bool entry_not_in_sparsity_pattern = false;
    assert(!entry_not_in_sparsity_pattern);
  }

 public:
  std::vector<MatrixXd>& supernodes_;
  std::vector<int>& residual_size_;
  std::vector<Clique>& path_;
  std::vector<MatrixXd>& separator_;
};

// Returns the supernode subindex and the seperator subindex for two overlapping
// cliques. 
using Match = std::pair<int, int>;
std::vector<Match> Intersection(const SparseTriangularMatrix& mat, int supernode, int seperator) {
  std::vector<Match> y;
  // Supernodes
  for (int i = 0; i < mat.residual_size.at(supernode); i++) {
    // Non-supernodes
    for (size_t j = mat.residual_size.at(seperator); j < mat.path.at(seperator).size(); j++) {
      if (mat.path.at(supernode).at(i) == mat.path.at(seperator).at(j)) {
        y.emplace_back(i, j  - mat.residual_size.at(seperator));
      }
    }
  }
  return y;
}

class PartitionVectorForwardIterator {
 public:
  PartitionVectorForwardIterator(VectorXd& b, const std::vector<int>& sizes) : b_(b), sizes_(sizes) {
    i_ = 0;
    size_i = sizes_.at(i_);
    start_i = 0; 
  }

  Eigen::Ref<VectorXd> b_i() { return b_.segment(start_i, size_i); }
  Eigen::Ref<VectorXd> b_i_minus_1() { return b_.segment(start_i_minus_1, size_i_minus_1); }
  void Increment()  {
    start_i_minus_1 = start_i;
    size_i_minus_1 = size_i;
    i_++;
    size_i = sizes_.at(i_);
    start_i = start_i_minus_1 + size_i_minus_1;
  }

  int i_ = 0;
  int start_i_minus_1; 
  int start_i; 
  int size_i_minus_1; 
  int size_i;
  VectorXd& b_;
  const std::vector<int>& sizes_;
  void Set(int i) {
    if (i > 0) {
      assert(0);
    }
    if (i < i_) {
      assert(0);
    }
    while (i > i_) {
      Increment();
    }
  }
};

class PartitionVectorIterator {
 public:
  PartitionVectorIterator(VectorXd& b, int N, const std::vector<int>& sizes) : b_(b), sizes_(sizes) {
    i_ = sizes.size() - 1;
    size_i = sizes_.at(i_);
    start_i = N - size_i;
  }

  Eigen::Ref<VectorXd> b_i() { return b_.segment(start_i, size_i); }
  Eigen::Ref<VectorXd> b_i_plus_1() { return b_.segment(start_i_plus_1, size_i_plus_1); }
  void Decrement()  {
    start_i_plus_1 = start_i;
    size_i_plus_1 = size_i;
    i_--;
    size_i = sizes_.at(i_);
    start_i = start_i_plus_1 - size_i;
  }

  int i_ = 0;
  int start_i_plus_1; 
  int start_i; 
  int size_i_plus_1; 
  int size_i;
  VectorXd& b_;
  const std::vector<int>& sizes_;
  void Set(int i) {
    if (i < 0) {
      assert(0);
    }
    if (i > i_) {
      assert(0);
    }
    while (i < i_) {
      Decrement();
    }
  }
};


}

//  Given block lower-triangular matrix, applies the recursion
//    c_1 c_2 c_3
//    L_1 B_2 B_2
//        L_2 B_1
//            L_3
//
//   y_{i} = inv(L_{i}) r_{i}
//   r = r -  c_i * y_i
//
//   Structure of B_i:  non-zero columns are dense. 
void T::ApplyBlockInverseOfTransposeInPlace(const SparseTriangularMatrix& mat, VectorXd* y) {
  PartitionVectorIterator ypart(*y, mat.N, mat.residual_size);
  mat.supernodes.back().triangularView<Eigen::Lower>().transpose().solveInPlace(ypart.b_i());

  for (int i = static_cast<int>(mat.supernodes.size() - 2); i >= 0; i--) {
    ypart.Decrement();

    for (int j = i; j >= 0; j--) {
      PartitionVectorIterator residual(*y, mat.N, mat.residual_size);
      residual.Set(j);
      // Loop over intersection of supernode i+1 and seperator j.
      auto index_and_column_list = Intersection(mat, i + 1, j);
      for (const auto& pair : index_and_column_list) {
        residual.b_i() -=  mat.separator.at(j).col(pair.second) * 
            ypart.b_i_plus_1()(pair.first);
      }
    }
    mat.supernodes.at(i).triangularView<Eigen::Lower>().transpose().solveInPlace(ypart.b_i());
  }
}

//  c_1 c_2 c_3
//  L_1 
//  B_1 L_2 
//  B_1 B_2 L_3
//
//   y_{i} = inv(L_{i}) r_{i}
//   r = r -  c_i * y_i
void T::ApplyBlockInverseInPlace(const SparseTriangularMatrix& mat, VectorXd* y) {

  PartitionVectorForwardIterator ypart(*y,  mat.residual_size);
  mat.supernodes.at(0).triangularView<Eigen::Lower>().solveInPlace(ypart.b_i());

  for (size_t i = 1; i < mat.supernodes.size(); i++) {
    ypart.Increment();
    if (mat.separator.at(i-1).size() > 0) {
      VectorXd temp =  mat.separator.at(i-1).transpose() * ypart.b_i_minus_1();
      int cnt = 0;
      for (size_t j = mat.residual_size.at(i-1); j < mat.path.at(i-1).size(); j++) {
        (*y)(mat.path.at(i-1).at(j)) -= temp(cnt);
        cnt++;
      }
    }

    mat.supernodes.at(i).triangularView<Eigen::Lower>().solveInPlace(ypart.b_i());
  }
}


void T::BlockCholeskyInPlace(SparseTriangularMatrix* C) {
  std::vector<Eigen::LLT<Eigen::Ref<MatrixXd>>> llts;
  for (size_t i = 0; i < C->supernodes.size(); i++) {
    llts.emplace_back(C->supernodes.at(i));
    C->supernodes.at(i).triangularView<Eigen::Lower>().solveInPlace(C->separator.at(i));
    auto& temp = C->separator.at(i);

    LowerTriangularSuperNodal mat(C);
    for (int k = 0; k < temp.cols(); k++) {
      for (int j = k; j < temp.cols(); j++) {
        int c = C->path.at(i).at(C->residual_size.at(i) + k);
        int r = C->path.at(i).at(C->residual_size.at(i) + j);
        mat.Increment(-temp.col(k).dot(temp.col(j)), r, c);
      }
    }
  }
}
