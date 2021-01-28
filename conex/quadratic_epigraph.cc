#include "conex/quadratic_cone_constraint.h"

namespace conex {
namespace {

DenseMatrix BuildQ(const DenseMatrix& Qi) {
  // Set inner-product matrix Q of Lorentz cone L.
  Eigen::MatrixXd Q(Qi.rows() + 1, Qi.rows() + 1);
  Q.setZero();
  Q(0, 0) = 1;
  Q.bottomRightCorner(Qi.rows(), Qi.rows()) = Qi;
  return Q;
}

DenseMatrix BuildA(const DenseMatrix& Q) {
  // Build (A, b) satisfying b - A(x, t) \in L <=> t >= 1/2 x^T Q x.
  int num_vars = Q.rows();
  Eigen::MatrixXd Ai(num_vars + 2, num_vars + 1);
  Ai.setZero();
  Ai.topRightCorner(2, 1) << -0.5, -0.5;
  Ai.bottomLeftCorner(num_vars, num_vars) =
      Eigen::MatrixXd::Identity(num_vars, num_vars);
  return Ai;
}

DenseMatrix Buildb(const DenseMatrix& Q) {
  int num_vars = Q.rows();
  Eigen::MatrixXd b(num_vars + 2, 1);
  b.setZero();
  b(0) = 1;
  b(1) = -1;
  return b;
}

}  // namespace

using T = QuadraticEpigraph;
using B = QuadraticConstraintBase;
T::QuadraticEpigraph(const DenseMatrix& Qi)
    : QuadraticConstraintBase(BuildQ(Qi), BuildA(Qi), Buildb(Qi)) {}

void T::Initialize() { B::Initialize(); }

// Apply operator:
//    0, Qi
//  -.5, 0
DenseMatrix T::EvalAtQX(const DenseMatrix& X, DenseMatrix* QX) {
  int n = Q_.rows() - 1;
  const auto& Qi = Q_.bottomRightCorner(n, n);
  DenseMatrix Y(n + 1, X.cols());
  Y.topRows(n) = Qi * X.bottomRows(n);
  Y.bottomRows(1) = -.5 * X.topRows(1);
  return Y;
}
DenseMatrix T::EvalAtQX(const DenseMatrix& X, Ref*) {
  int n = Q_.rows() - 1;
  const auto& Qi = Q_.bottomRightCorner(n, n);
  DenseMatrix Y(n + 1, X.cols());
  Y.topRows(n) = Qi * X.bottomRows(n);
  Y.bottomRows(1) = -.5 * X.topRows(1);
  return Y;
}

}  // namespace conex
