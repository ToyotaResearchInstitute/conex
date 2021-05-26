#pragma once

#include <vector>
#include "kkt_assembler.h"
#include <Eigen/Dense>

namespace conex {

struct WorkspaceEqualityConstraints {
  using DenseMatrix = Eigen::MatrixXd;

  friend int SizeOf(const WorkspaceEqualityConstraints& o) { return 0; }

  friend void Initialize(WorkspaceEqualityConstraints*, double*) {}

  friend void print(const WorkspaceEqualityConstraints&) {}
  Eigen::Map<DenseMatrix, Eigen::Aligned> W{NULL, 0, 0};
};

class EqualityConstraints : public LinearKKTAssemblerBase {
 public:
  EqualityConstraints(){};
  EqualityConstraints(const Eigen::MatrixXd& A, const Eigen::MatrixXd& b)
      : A_(A), b_(b), lambda_(Eigen::VectorXd::Zero(b_.rows())) {}

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
  Eigen::VectorXd lambda_;

  friend int Rank(const EqualityConstraints&) { return 0; };
  friend void SetIdentity(EqualityConstraints*){};
  friend void PrepareStep(EqualityConstraints* o, const StepOptions&,
                          const Ref& y, StepInfo*) {
    o->lambda_ = y.col(0).tail(o->b_.rows());
  }
  friend bool TakeStep(EqualityConstraints*, const StepOptions&) {
    return true;
  };
  friend void GetWeightedSlackEigenvalues(EqualityConstraints*, const Ref&,
                                          WeightedSlackEigenvalues*){};

  friend void ConstructSchurComplementSystem(EqualityConstraints* o,
                                             bool initialize,
                                             SchurComplementSystem* sys_) {
    auto& sys = *sys_;
    auto& A_ = o->A_;
    auto& b_ = o->b_;
    if (initialize) {
      sys.G.setZero();
      sys.G.bottomLeftCorner(A_.rows(), A_.cols()) = A_;
      sys.G.topRightCorner(A_.cols(), A_.rows()) = A_.transpose();
      sys.AQc.setZero();
      sys.AQc.bottomRows(A_.rows()) = b_;
      sys.AW.setZero();
      sys.inner_product_of_w_and_c = o->lambda_.dot(b_.col(0));
    } else {
      sys.G.bottomLeftCorner(A_.rows(), A_.cols()) += A_;
      sys.G.topRightCorner(A_.cols(), A_.rows()) += A_.transpose();
      sys.AQc.bottomRows(A_.rows()) += b_;
      sys.inner_product_of_w_and_c += o->lambda_.dot(b_.col(0));
    }
  }

  int number_of_variables() { return 0; }
  WorkspaceEqualityConstraints workspace_;
  WorkspaceEqualityConstraints* workspace() { return &workspace_; }
};

}  // namespace conex
