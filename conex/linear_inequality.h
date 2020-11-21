#include <vector>
#include <Eigen/Dense>
#include "conex/supernodal_cholesky_data.h"

class LinearInequality {
 public:
  LinearInequality(const Eigen::MatrixXd& A) :
      A_(A) { 
   for (int i = 0; i < A.cols(); i++) {
    variables_.push_back(i);
   }
  }

  void SetPartition(const std::vector<int>& supernodes,
                    const std::vector<int>& separators) {
    separators_ = separators;
    supernodes_ = supernodes; 
  }

  void SetOffDiagonal(Eigen::Map<Eigen::MatrixXd>* data);
  void SetSupernodeDiagonal(Eigen::Map<Eigen::MatrixXd>* data);
  void IncrementSeparatorDiagonal(Eigen::Map<Eigen::MatrixXd>* data) {};

  void BindDiagonalBlock(const DiagonalBlock* data);
  void BindOffDiagonalBlock(const OffDiagonalBlock* data);
  void UpdateBlocks();
  int GetCoeff(int i, int j);

 private:
  void Increment(std::vector<int> r, std::vector<int> c, Eigen::Map<Eigen::MatrixXd>* data);
  void Set(std::vector<int> r, std::vector<int> c, Eigen::Map<Eigen::MatrixXd>* data);
  void Scatter(const std::vector<int>& r, const std::vector<int>& c, double** data);
  Eigen::MatrixXd A_;
  std::vector<int> variables_;
  
  std::vector<int> supernodes_;
  std::vector<int> separators_;

  std::vector<DiagonalBlock> diag;
  std::vector<OffDiagonalBlock> off_diag;
  std::vector<OffDiagonalBlock> scatter_block;
};

