#include "conex/equality_constraint.h"
#include "conex/debug_macros.h"

namespace conex {

using T = EqualityConstraints;
using Eigen::MatrixXd;
using std::vector;

T::EqualityConstraints(const Eigen::MatrixXd& A, const Eigen::MatrixXd& b)
    : A_(A), b_(b), lambda_(Eigen::VectorXd::Zero(b_.rows())) {}
// void T::SetDenseData() {
//  DUMP("IN DENSE!");
//  assert(schur_complement_data.G.rows() == schur_complement_data.G.cols());
//  assert(schur_complement_data.G.rows() == A_.rows() + A_.cols());
//
//  schur_complement_data.G.setZero();
//  schur_complement_data.G.bottomLeftCorner(A_.rows(), A_.cols()) = A_;
//  schur_complement_data.G.topRightCorner(A_.cols(), A_.rows()) =
//      A_.transpose();
//  schur_complement_data.AQc.bottomRows(A_.rows()) = b_;
//  schur_complement_data.AW.setZero();
//  DUMP(schur_complement_data.G);
//}

void ConstructSchurComplementSystem(EqualityConstraints* o, bool initialize,
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

}  // namespace conex
