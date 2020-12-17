#pragma once

#include <vector>
#include "conex/supernodal_cholesky_data.h"
#include "kkt_assembler.h"
#include <Eigen/Dense>

namespace conex {

class EqualityConstraints : public LinearKKTAssemblerBase {
 public:
  EqualityConstraints(){};
  EqualityConstraints(const Eigen::MatrixXd& A, const Eigen::MatrixXd& b)
      : A_(A), b_(b) {}

  virtual void SetDenseData() override {
    assert(schur_complement_data.G.rows() == schur_complement_data.G.cols());
    assert(schur_complement_data.G.rows() == A_.rows() + A_.cols());

    schur_complement_data.G.setZero();
    schur_complement_data.G.bottomLeftCorner(A_.rows(), A_.cols()) = A_;
    schur_complement_data.G.topRightCorner(A_.cols(), A_.rows()) =
        A_.transpose();
    schur_complement_data.AQc.bottomRows(A_.rows()) = b_;
    schur_complement_data.AW.setZero();
  }
  int SizeOfDualVariable() { return A_.rows(); }
  Eigen::MatrixXd A_;
  Eigen::MatrixXd b_;
};

}  // namespace conex
