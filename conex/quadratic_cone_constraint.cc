#include "conex/quadratic_cone_constraint.h"
#include "conex/newton_step.h"

namespace conex {

using EigenType = DenseMatrix;
using Real = double;

namespace {

// double InnerProduct(const DenseMatrix& Q, double x0, const Eigen::VectorXd&
// x1,
//                    const double y0, const Eigen::VectorXd& y1) {
//  return 2 * (x0 * y0 + x1.transpose() * Q * y1);
//}

double SquaredNorm(const DenseMatrix& Q, const Eigen::VectorXd& x) {
  if (Q.rows() > 0)
    return x.transpose() * Q * x;
  else {
    return x.transpose() * x;
  }
}

double Norm(const Eigen::MatrixXd& Q, const Eigen::VectorXd& x) {
  return std::sqrt(SquaredNorm(Q, x));
}

struct SpinFactorElement {
  double q0;
  Eigen::VectorXd q1;
};

double InnerProduct(const DenseMatrix& Q, const Eigen::VectorXd& x,
                    const Eigen::VectorXd& y) {
  if (Q.rows() > 0)
    return x.transpose() * Q * y;
  else {
    return x.transpose() * y;
  }
}

SpinFactorElement QuadraticRepresentation(double x1_norm_squared,
                                          double inner_product_of_x1_and_y1,
                                          double x0, const Eigen::VectorXd& x1,
                                          double y0,
                                          const Eigen::VectorXd& y1) {
  // We use the formula from Example 11.12 of "Formally Real Jordan Algebras
  // and Their Applications to Optimization"  by Alizadeh, which states the
  // quadratic representation of x equals the linear map
  //                          2xx' - (det x) * R
  // where R is the reflection operator R = diag(1, -1, ..., -1) and det x is
  // the determinate of x = (x0, x1), i.e., det x = x0^2 - |x1|^2.
  double det_x = x0 * x0 - x1_norm_squared;
  SpinFactorElement z;
  double scale = 2 * (x0 * y0 + inner_product_of_x1_and_y1);
  z.q0 = scale * x0 - det_x * y0;
  z.q1 = scale * x1 + det_x * y1;
  return z;
}

void Exp(double norm_x1, double* x0, Eigen::VectorXd* x1) {
  double k = norm_x1;
  if (k > 0) {
    (*x1) *= .5 * (exp(*x0 + k) - exp(*x0 - k)) / k;
  }
  (*x0) = (.5 * (exp(*x0 + k) + exp(*x0 - k)));
}

template <typename T>
void Sqrt(double norm_x1, double* x0, T* x1) {
  double k = norm_x1;
  if (k > 0) {
    (*x1) *= .5 * (std::sqrt(*x0 + k) - std::sqrt(*x0 - k)) / k;
  }
  (*x0) = (.5 * (std::sqrt(*x0 + k) + std::sqrt(*x0 - k)));
}

DenseMatrix Eigenvalues(double norm_of_x1, double x0) {
  DenseMatrix eigenvalues(2, 1);
  eigenvalues(0) = x0 + norm_of_x1;
  eigenvalues(1) = x0 - norm_of_x1;
  return eigenvalues;
}

}  // namespace

void QuadraticConstraint::ComputeNegativeSlack(double inv_sqrt_mu, const Ref& y,
                                               Ref* minus_s) {
  int n = A1_.rows();
  (*minus_s)(0, 0) = A0_.dot(y.col(0));
  (*minus_s)(0, 0) -= C0_ * inv_sqrt_mu;
  minus_s->bottomRows(n).noalias() = A1_ * y;
  minus_s->bottomRows(n).noalias() -= C1_ * inv_sqrt_mu;
}

// Combine this with TakeStep
void GetMuSelectionParameters(QuadraticConstraint* o, const Ref& y,
                              MuSelectionParameters* p) {
  auto* workspace = &o->workspace_;
  auto& minus_s = workspace->temp_1;
  SpinFactorElement Ws;
  o->ComputeNegativeSlack(1, y, &minus_s);

  SpinFactorElement wsqrt;
  wsqrt.q0 = o->workspace_.W0;
  wsqrt.q1 = o->workspace_.W1;

  Sqrt(Norm(o->Q_, wsqrt.q1), &wsqrt.q0, &wsqrt.q1);

  int n = workspace->n_;
  Ws = QuadraticRepresentation(
      SquaredNorm(o->Q_, wsqrt.q1),
      InnerProduct(o->Q_, wsqrt.q1, minus_s.bottomRows(n)), wsqrt.q0, wsqrt.q1,
      minus_s(0, 0), minus_s.bottomRows(n));

  auto ev = Eigenvalues(Norm(o->Q_, Ws.q1), Ws.q0);

  const double lamda_max = -ev.minCoeff();
  const double lamda_min = -ev.maxCoeff();

  if (p->gw_lambda_max < lamda_max) {
    p->gw_lambda_max = lamda_max;
  }
  if (p->gw_lambda_min > lamda_min) {
    p->gw_lambda_min = lamda_min;
  }
  p->gw_norm_squared += std::pow(lamda_max, 2) + std::pow(lamda_min, 2);
  p->gw_trace += (lamda_max + lamda_min);
}

void TakeStep(QuadraticConstraint* o, const StepOptions& opt, const Ref& y,
              StepInfo* info) {
  auto& minus_s = o->workspace_.temp_1;
  o->ComputeNegativeSlack(opt.inv_sqrt_mu, y, &minus_s);

  // d =  e - Q(w^{1/2})(C-A^y)
  double wsqrt_q0 = o->workspace_.W0;
  Eigen::VectorXd wsqrt_q1 = o->workspace_.W1;
  Sqrt(Norm(o->Q_, wsqrt_q1), &wsqrt_q0, &wsqrt_q1);

  int n = wsqrt_q1.rows() + 1;
  auto d = QuadraticRepresentation(
      SquaredNorm(o->Q_, wsqrt_q1),
      InnerProduct(o->Q_, wsqrt_q1, minus_s.bottomRows(n - 1)), wsqrt_q0,
      wsqrt_q1, minus_s(0, 0), minus_s.bottomRows(n - 1));
  d.q0 += 1;

  // Compute rescaling.
  auto ev = Eigenvalues(Norm(o->Q_, d.q1), d.q0);
  info->norminfd = std::fabs(ev(0));
  if (info->norminfd < std::fabs(ev(1))) {
    info->norminfd = std::fabs(ev(1));
  }
  info->normsqrd = ev.squaredNorm();

  double scale = info->norminfd * info->norminfd;
  if (scale > 2.0) {
    d.q0 = 2 * d.q0 / scale;
    d.q1 = 2 * d.q1 / scale;
  }

  Exp(Norm(o->Q_, d.q1), &d.q0, &d.q1);
  const auto& expd = d;

  auto wn = QuadraticRepresentation(SquaredNorm(o->Q_, wsqrt_q1),
                                    InnerProduct(o->Q_, wsqrt_q1, expd.q1),
                                    wsqrt_q0, wsqrt_q1, expd.q0, expd.q1);
  o->workspace_.W0 = wn.q0;
  o->workspace_.W1 = wn.q1;
}

template <typename T>
void SchurComplement(const Eigen::VectorXd& A0, const Eigen::MatrixXd& A_gram,
                     const double W0, double det_w,
                     const Eigen::MatrixXd& A_dot_w, bool initialize, T* G) {
  if (initialize) {
    *G = det_w * (A_gram - A0 * A0.transpose());
  } else {
    *G += det_w * (A_gram - A0 * A0.transpose());
  }
  (*G) += 2 * (A_dot_w + A0 * W0) * (A_dot_w + A0 * W0).transpose();
}

DenseMatrix EvalAtQX(const DenseMatrix& A, const DenseMatrix& Q,
                     const DenseMatrix& X) {
  if (Q.rows() > 0) {
    return A.transpose() * Q * X;
  } else {
    return A.transpose() * X;
  }
}

void QuadraticConstraint::Initialize() { A_gram_ = EvalAtQX(A1_, Q_, A1_); }

void ConstructSchurComplementSystem(QuadraticConstraint* o, bool initialize,
                                    SchurComplementSystem* sys) {
  const Eigen::VectorXd& A0 = o->A0_;
  const Eigen::MatrixXd& A1 = o->A1_;
  const auto& C0 = o->C0_;
  const auto& C1 = o->C1_;
  const Eigen::MatrixXd& A_gram = o->A_gram_;
  const Eigen::MatrixXd& A_dot_x = EvalAtQX(A1, o->Q_, o->workspace_.W1);

  double det_w = o->workspace_.W0 * o->workspace_.W0 -
                 SquaredNorm(o->Q_, o->workspace_.W1);

  if (initialize) {
    SchurComplement(A0, A_gram, o->workspace_.W0, det_w, A_dot_x, true,
                    &sys->G);
    sys->AW.noalias() = A_dot_x + A0 * o->workspace_.W0;
    sys->AQc.noalias() = det_w * (EvalAtQX(A1, o->Q_, C1) - A0 * C0);

  } else {
    SchurComplement(A0, A_gram, o->workspace_.W0, det_w, A_dot_x, false,
                    &sys->G);
    sys->AW.noalias() += A_dot_x + A0 * o->workspace_.W0;
    sys->AQc.noalias() += det_w * (EvalAtQX(A1, o->Q_, C1) - A0 * C0);
  }
  double scale =
      InnerProduct(o->Q_, C1, o->workspace_.W1) + C0 * o->workspace_.W0;
  sys->AQc.noalias() += 2 * (A_dot_x + A0 * o->workspace_.W0) * scale;
}

}  // namespace conex
