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

template <typename T>
double SquaredNorm(const DenseMatrix& Q, const Eigen::VectorXd& x,
                   T* workspace) {
  if (Q.rows() > 0) {
    *workspace = Q * x;
    return x.dot(workspace->col(0));
  } else {
    return x.dot(x);
  }
}

template <typename T>
double Norm(const Eigen::MatrixXd& Q, const Eigen::VectorXd& x, T* workspace) {
  return std::sqrt(SquaredNorm(Q, x, workspace));
}

template <typename T>
double InnerProduct(const DenseMatrix& Q, const Eigen::VectorXd& x,
                    const Eigen::VectorXd& y, T* workspace) {
  if (Q.rows() > 0) {
    *workspace = Q * y;
    return x.dot(workspace->col(0));
  } else {
    return x.transpose() * y;
  }
}

template <typename T>
void QuadraticRepresentation(double x1_norm_squared,
                             double inner_product_of_x1_and_y1, double x0,
                             const Eigen::VectorXd& x1, double y0,
                             const Eigen::VectorXd& y1, double* z_q0, T* z_q1) {
  // We use the formula from Example 11.12 of "Formally Real Jordan Algebras
  // and Their Applications to Optimization"  by Alizadeh, which states the
  // quadratic representation of x equals the linear map
  //                          2xx' - (det x) * R
  // where R is the reflection operator R = diag(1, -1, ..., -1) and det x is
  // the determinate of x = (x0, x1), i.e., det x = x0^2 - |x1|^2.
  double det_x = x0 * x0 - x1_norm_squared;
  double scale = 2 * (x0 * y0 + inner_product_of_x1_and_y1);
  *z_q0 = scale * x0 - det_x * y0;
  *z_q1 = scale * x1 + det_x * y1;
}

template <typename T>
void Exp(double norm_x1, double* x0, T* x1) {
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

Eigen::Vector2d Eigenvalues(double norm_of_x1, double x0) {
  Eigen::Vector2d eigenvalues(2, 1);
  eigenvalues(0) = x0 + norm_of_x1;
  eigenvalues(1) = x0 - norm_of_x1;
  return eigenvalues;
}

}  // namespace

void QuadraticConstraint::ComputeNegativeSlack(double inv_sqrt_mu, const Ref& y,
                                               double* minus_s_0,
                                               Ref* minus_s_1) {
  *minus_s_0 = A0_.dot(y.col(0));
  *minus_s_0 -= C0_ * inv_sqrt_mu;
  minus_s_1->noalias() = A1_ * y;
  minus_s_1->noalias() -= C1_ * inv_sqrt_mu;
}

// Combine this with TakeStep
void GetMuSelectionParameters(QuadraticConstraint* o, const Ref& y,
                              MuSelectionParameters* p) {
  auto* workspace = &o->workspace_;
  auto& minus_s_1 = workspace->temp1_1;
  double minus_s_0;
  o->ComputeNegativeSlack(1, y, &minus_s_0, &minus_s_1);

  double wsqrt_q0 = o->workspace_.W0;
  auto& wsqrt_q1 = o->workspace_.temp3_1;
  wsqrt_q1 = o->workspace_.W1;

  Sqrt(Norm(o->Q_, wsqrt_q1, &workspace->temp2_1), &wsqrt_q0, &wsqrt_q1);

  auto& Ws_1 = workspace->temp2_1;
  double Ws_0;
  QuadraticRepresentation(
      SquaredNorm(o->Q_, wsqrt_q1, &workspace->temp2_1),
      InnerProduct(o->Q_, wsqrt_q1, minus_s_1, &workspace->temp2_1), wsqrt_q0,
      wsqrt_q1, minus_s_0, minus_s_1, &Ws_0, &Ws_1);

  auto ev = Eigenvalues(Norm(o->Q_, Ws_1, &workspace->temp1_1), Ws_0);

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
  // d =  e - Q(w^{1/2})(C-A^y)
  double wsqrt_q0 = o->workspace_.W0;
  auto& wsqrt_q1 = o->workspace_.temp3_1;
  wsqrt_q1 = o->workspace_.W1;
  auto& d_q1 = o->workspace_.temp2_1;
  double d_q0;

  {  // Use temp_1
    auto& minus_s_1 = o->workspace_.temp1_1;
    double minus_s_0;
    o->ComputeNegativeSlack(opt.inv_sqrt_mu, y, &minus_s_0, &minus_s_1);

    Sqrt(Norm(o->Q_, wsqrt_q1, &o->workspace_.temp2_1), &wsqrt_q0, &wsqrt_q1);

    QuadraticRepresentation(
        SquaredNorm(o->Q_, wsqrt_q1, &o->workspace_.temp2_1),
        InnerProduct(o->Q_, wsqrt_q1, minus_s_1, &o->workspace_.temp2_1),
        wsqrt_q0, wsqrt_q1, minus_s_0, minus_s_1, &d_q0, &d_q1);
    d_q0 += 1;
  }  // temp_1 is free

  // Compute rescaling.
  auto ev = Eigenvalues(Norm(o->Q_, d_q1, &o->workspace_.temp1_1), d_q0);
  info->norminfd = std::fabs(ev(0));
  if (info->norminfd < std::fabs(ev(1))) {
    info->norminfd = std::fabs(ev(1));
  }
  info->normsqrd = ev.squaredNorm();

  double scale = info->norminfd * info->norminfd;
  if (scale > 2.0) {
    d_q0 = 2 * d_q0 / scale;
    d_q1 = 2 * d_q1 / scale;
  }

  Exp(Norm(o->Q_, d_q1, &o->workspace_.temp1_1), &d_q0, &d_q1);
  const auto& expd_q1 = d_q1;
  const auto& expd_q0 = d_q0;

  QuadraticRepresentation(
      SquaredNorm(o->Q_, wsqrt_q1, &o->workspace_.temp1_1),
      InnerProduct(o->Q_, wsqrt_q1, expd_q1, &o->workspace_.temp1_1), wsqrt_q0,
      wsqrt_q1, expd_q0, expd_q1, &o->workspace_.W0, &o->workspace_.W1);
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
                 SquaredNorm(o->Q_, o->workspace_.W1, &o->workspace_.temp1_1);

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
      InnerProduct(o->Q_, C1, o->workspace_.W1, &o->workspace_.temp1_1) +
      C0 * o->workspace_.W0;
  sys->AQc.noalias() += 2 * (A_dot_x + A0 * o->workspace_.W0) * scale;
}

}  // namespace conex
