#include "conex/block_triangular_operations.h"
#include "conex/RLDLT.h"
#include "conex/debug_macros.h"

namespace conex {

using Eigen::MatrixXd;
using Eigen::VectorXd;

using T = BlockTriangularOperations;
namespace {

class PartitionVectorForwardIterator {
 public:
  PartitionVectorForwardIterator(VectorXd& b, const std::vector<int>& sizes)
      : b_(b), sizes_(sizes) {
    Reset();
  }

  Eigen::Ref<VectorXd> b_i() { return b_.segment(start_i, size_i); }
  Eigen::Ref<VectorXd> b_i_minus_1() {
    return b_.segment(start_i_minus_1, size_i_minus_1);
  }

  void Reset() {
    i_ = 0;
    size_i = sizes_.at(i_);
    start_i = 0;
  }
  void Increment() {
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
  PartitionVectorIterator(VectorXd& b, int N, const std::vector<int>& sizes)
      : b_(b), N_(N), sizes_(sizes) {
    Reset();
  }

  Eigen::Ref<VectorXd> b_i() { return b_.segment(start_i, size_i); }
  Eigen::Ref<VectorXd> b_i_plus_1() {
    return b_.segment(start_i_plus_1, size_i_plus_1);
  }
  void Reset() {
    i_ = sizes_.size() - 1;
    size_i = sizes_.at(i_);
    start_i = N_ - size_i;
  }
  void Decrement() {
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
  const int N_;
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
}  // namespace

//  Given block lower-triangular matrix, applies the recursion
//    c_1 c_2 c_3
//    L_1 B_2 B_2
//        L_2 B_1
//            L_3
//
//   y_{i} = inv(L_{i}) b_{i}
//   b = b -  c_i * y_i
//
//   Structure of B_i:  non-zero columns are dense.
void T::ApplyBlockInverseOfTransposeInPlace(
    const TriangularMatrixWorkspace& mat, VectorXd* y) {
  PartitionVectorIterator ypart(*y, mat.N, mat.supernode_size);
  mat.diagonal.back()
      .triangularView<Eigen::Lower>()
      .solveInPlace<Eigen::OnTheRight>(ypart.b_i().transpose());

  // Loop over partition {B_j} of c_{i+1}
  PartitionVectorIterator residual(*y, mat.N, mat.supernode_size);
  for (int i = static_cast<int>(mat.diagonal.size() - 2); i >= 0; i--) {
    ypart.Decrement();
    residual.Reset();

    int jcnt = 0;
    for (auto j : mat.column_intersections.at(i)) {
      residual.Set(j);
      // Find columns of B_j that are nonzero on columns c_{i+1} of supernode
      // i+1. This corresponds to separators(i) that contain supernode(j) for j
      // > i.
      const auto& index_and_column_list =
          mat.intersection_position.at(i).at(jcnt++);
      for (const auto& pair : index_and_column_list) {
        residual.b_i() -= mat.off_diagonal.at(j).col(pair.second) *
                          ypart.b_i_plus_1()(pair.first);
      }
    }

    mat.diagonal.at(i).triangularView<Eigen::Lower>().transpose().solveInPlace(
        ypart.b_i());
  }
}

//  c_1 c_2 c_3
//  L_1
//  B_1 L_2
//  B_1 B_2 L_3
//
//   y_{i} = inv(L_{i}) r_{i}
//   r = r -  c_i * y_i
void T::ApplyBlockInverseInPlace(const TriangularMatrixWorkspace& mat,
                                 VectorXd* y) {
  PartitionVectorForwardIterator ypart(*y, mat.supernode_size);
  mat.diagonal.at(0).triangularView<Eigen::Lower>().solveInPlace(ypart.b_i());

  for (size_t i = 1; i < mat.diagonal.size(); i++) {
    ypart.Increment();
    if (mat.off_diagonal.at(i - 1).size() > 0) {
      VectorXd temp =
          mat.off_diagonal.at(i - 1).transpose() * ypart.b_i_minus_1();
      int cnt = 0;
      for (auto si : mat.separators.at(i - 1)) {
        (*y)(si) -= temp(cnt);
        cnt++;
      }
    }
    mat.diagonal.at(i).triangularView<Eigen::Lower>().solveInPlace(ypart.b_i());
  }
}

bool T::BlockCholeskyInPlace(TriangularMatrixWorkspace* C) {
  assert(C->diagonal.size() == C->off_diagonal.size());
  std::vector<Eigen::LLT<Eigen::Ref<MatrixXd>>> llts;
  for (size_t i = 0; i < C->diagonal.size(); i++) {
    // In place LLT of [n, n] block
    llts.emplace_back(C->diagonal.at(i));
    if (llts.back().info() != Eigen::Success) {
      return false;
    }

    // Construction of [n, s] block
    if (C->off_diagonal.at(i).size() > 0) {
      llts.back().matrixL().solveInPlace(C->off_diagonal.at(i));
      auto& temp = C->off_diagonal.at(i);

      int index = 0;
      const auto& s_s = C->seperator_diagonal.at(i);
      for (int k = 0; k < temp.cols(); k++) {
        for (int j = k; j < temp.cols(); j++) {
          *s_s.at(index++) -= temp.col(k).dot(temp.col(j));
        }
      }
    }
  }
  return true;
}

// Apply inv(M^T)  = inv(L^T P) = P^T inv(L^T)
void T::ApplyBlockInverseOfMTranspose(
    const TriangularMatrixWorkspace& mat,
    const std::vector<Eigen::RLDLT<Eigen::Ref<MatrixXd>>> factorization,
    VectorXd* y) {
  PartitionVectorIterator ypart(*y, mat.N, mat.supernode_size);
  // mat.diagonal.back().triangularView<Eigen::Lower>().transpose().solveInPlace(ypart.b_i());
  factorization.back().matrixL().transpose().solveInPlace(ypart.b_i());
  Eigen::PermutationMatrix<-1> P0(factorization.back().transpositionsP());
  ypart.b_i() = P0.transpose() * ypart.b_i();

  for (int i = static_cast<int>(mat.diagonal.size() - 2); i >= 0; i--) {
    ypart.Decrement();

    // Loop over partition {B_j} of c_{i+1}
    PartitionVectorIterator residual(*y, mat.N, mat.supernode_size);

    int jcnt = 0;
    for (auto j : mat.column_intersections.at(i)) {
      residual.Set(j);
      // Find columns of B_j that are nonzero on columns c_{i+1} of supernode
      // i+1. This corresponds to separators(i) that contain supernode(j) for j
      // > i.
      // auto index_and_column_list =
      //    IntersectionOfSupernodeAndSeparator(mat, i + 1, j);
      // for (const auto& pair : index_and_column_list) {
      //  residual.b_i() -= mat.off_diagonal.at(j).col(pair.second) *
      //                    ypart.b_i_plus_1()(pair.first);
      //}

      auto index_and_column_list = mat.intersection_position.at(i).at(jcnt++);
      for (const auto& pair : index_and_column_list) {
        residual.b_i() -= mat.off_diagonal.at(j).col(pair.second) *
                          ypart.b_i_plus_1()(pair.first);
      }
    }

    // mat.diagonal.at(i).triangularView<Eigen::Lower>().transpose().solveInPlace(ypart.b_i());
    factorization.at(i).matrixL().transpose().solveInPlace(ypart.b_i());
    Eigen::PermutationMatrix<-1> Pi(factorization.at(i).transpositionsP());
    ypart.b_i() = Pi.transpose() * ypart.b_i();
  }
}

void T::ApplyBlockInverseOfMD(
    const TriangularMatrixWorkspace& mat,
    const std::vector<Eigen::RLDLT<Eigen::Ref<MatrixXd>>> factorization,
    VectorXd* y) {
  // Apply inv(M) = inv(P^T L) = inv(L) P
  PartitionVectorForwardIterator ypart(*y, mat.supernode_size);
  Eigen::PermutationMatrix<-1> P0(factorization.at(0).transpositionsP());
  ypart.b_i() = P0 * ypart.b_i();
  factorization.at(0).matrixL().solveInPlace(ypart.b_i());

  for (size_t i = 1; i < mat.diagonal.size(); i++) {
    ypart.Increment();
    if (mat.off_diagonal.at(i - 1).size() > 0) {
      VectorXd temp =
          mat.off_diagonal.at(i - 1).transpose() * ypart.b_i_minus_1();
      int cnt = 0;
      for (auto si : mat.separators.at(i - 1)) {
        (*y)(si) -= temp(cnt);
        cnt++;
      }
    }
    Eigen::PermutationMatrix<-1> Pi(factorization.at(i).transpositionsP());
    ypart.b_i() = Pi * ypart.b_i();
    factorization.at(i).matrixL().solveInPlace(ypart.b_i());
  }

  // Apply D inverse
  PartitionVectorForwardIterator ypart2(*y, mat.supernode_size);
  ypart2.b_i() =
      factorization.at(0).vectorD().cwiseInverse().cwiseProduct(ypart2.b_i());
  for (size_t i = 1; i < mat.diagonal.size(); i++) {
    ypart2.Increment();
    ypart2.b_i() =
        factorization.at(i).vectorD().cwiseInverse().cwiseProduct(ypart2.b_i());
  }
}

// M D M^T
// M = P^T L
//
//   M    0   D_1      M^T   Q^T
//   Q    T       D_2        T^T
//
//  M D_1             M^T   Q^T
//  Q D_1     T D_2         T^T
//
//  [M D_1 M^T   M D_1 Q^T
//   Q D_1 M^T   Q D_1 Q^T + T D_2 T^T]
//
//  So, Q^T = inv(D_1) inv(M) off_diag
//          = inv(D_1) inv(L) P  * off_diag
bool T::BlockLDLTInPlace(
    TriangularMatrixWorkspace* C,
    std::vector<Eigen::RLDLT<Eigen::Ref<MatrixXd>>>* factorization) {
  auto& llts = *factorization;
  llts.clear();

  for (size_t i = 0; i < C->diagonal.size(); i++) {
    // In place LLT of [n, n] block
    llts.emplace_back(C->diagonal.at(i));
    if (llts.back().info() != Eigen::Success) {
      return false;
    }
    Eigen::PermutationMatrix<-1> P(llts.at(i).transpositionsP());

    //   Q^T = inv(D_1) inv(L) inv(P)  * off_diag
    if (C->off_diagonal.at(i).size() > 0) {
      C->off_diagonal.at(i) = P * C->off_diagonal.at(i);
      llts.back().matrixL().solveInPlace(C->off_diagonal.at(i));
      C->off_diagonal.at(i).noalias() =
          llts.back().vectorD().asDiagonal().inverse() *
          (C->off_diagonal.at(i));

      MatrixXd temp =
          llts.back().vectorD().asDiagonal() * C->off_diagonal.at(i);

      int index = 0;
      const auto& s_s = C->seperator_diagonal.at(i);
      for (int k = 0; k < temp.cols(); k++) {
        for (int j = k; j < temp.cols(); j++) {
          *s_s.at(index++) -= temp.col(k).dot(C->off_diagonal.at(i).col(j));
        }
      }
    }
  }
  return true;
}

}  // namespace conex
