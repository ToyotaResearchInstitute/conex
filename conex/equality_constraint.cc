#include "conex/equality_constraint.h"
#include "conex/debug_macros.h"

namespace conex {

using T = EqualityConstraints;
using Eigen::MatrixXd;
using std::vector;

T::EqualityConstraints(const Eigen::MatrixXd& A, const Eigen::MatrixXd& b)
    : A_(A), b_(b), lambda_(Eigen::VectorXd::Zero(b_.rows())) {}

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
