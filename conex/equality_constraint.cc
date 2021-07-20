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
    sys.inner_product_of_w_and_c = 0;
    sys.inner_product_of_c_and_Qc = 0;
  } else {
    sys.G.bottomLeftCorner(A_.rows(), A_.cols()) += A_;
    sys.G.topRightCorner(A_.cols(), A_.rows()) += A_.transpose();
    sys.AQc.bottomRows(A_.rows()) += b_;
  }
}

void PrepareStep(EqualityConstraints* o, const StepOptions&, const Ref& y,
                 StepInfo* info_i) {
  o->lambda_ = y.col(0).tail(o->b_.rows());
  info_i->normsqrd = 0;
  info_i->norminfd = 0;
}

}  // namespace conex
