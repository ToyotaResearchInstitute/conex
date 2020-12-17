#include <vector>
#include "conex/supernodal_cholesky_data.h"
#include <Eigen/Dense>

namespace conex {

class EqualityConstraints {
 public:
  EqualityConstraints(const Eigen::MatrixXd& A) : A_(A) {
    // The indices of each variable in the KKT System.
    //  0   A^T
    //  A   0
    for (int i = 0; i < A_.cols(); i++) {
      variables_.push_back(i);
    }
    for (int i = A_.cols(); i < A_.rows() + A_.cols(); i++) {
      dual_variables_.push_back(i);
    }
  }

  void SetPartition(const std::vector<int>& supernodes,
                    const std::vector<int>& separators) {
    separators_ = separators;
    supernodes_ = supernodes;
  }

  void SetOffDiagonal(Eigen::Map<Eigen::MatrixXd>* data);
  void SetSupernodeDiagonal(Eigen::Map<Eigen::MatrixXd>* data);
  void IncrementSeparatorDiagonal(Eigen::Map<Eigen::MatrixXd>* data){/*No OP*/};

  void BindDiagonalBlock(const DiagonalBlock* data);
  void BindOffDiagonalBlock(const OffDiagonalBlock* data);
  void UpdateBlocks();

  int SizeOfDualVariable() { return A_.rows(); }

 private:
  void Increment(std::vector<int> r, std::vector<int> c,
                 Eigen::Map<Eigen::MatrixXd>* data);
  void Set(std::vector<int> r, std::vector<int> c,
           Eigen::Map<Eigen::MatrixXd>* data);
  int GetCoeff(int i, int j);
  std::vector<int> variables_;
  std::vector<int> dual_variables_;

  std::vector<int> supernodes_;
  std::vector<int> separators_;
  Eigen::MatrixXd A_;

  bool cached_supernode_off_diagonal = false;
  Eigen::MatrixXd A_supernode_off_diagonal_;

  std::vector<DiagonalBlock> diag;
  std::vector<OffDiagonalBlock> off_diag;
};

}  // namespace conex
