#define EIGEN_RUNTIME_NO_MALLOC

#include "conex/quadratic_cone_constraint.h"
#include "conex/newton_step.h"

namespace conex {

using EigenType = DenseMatrix;
using Real = double;

namespace {

template <typename T1, typename T2, typename T3>
double SquaredNorm(const T1& Q, const T2& x, T3* workspace) {
  if (Q.rows() > 0) {
    workspace->noalias() = Q * x;
    return x.col(0).dot(workspace->col(0));
  } else {
    return x.col(0).dot(x.col(0));
  }
}

template <typename T1, typename T2, typename T3>
double Norm(const T1& Q, const T2& x, T3* workspace) {
  return std::sqrt(SquaredNorm(Q, x, workspace));
}

template <typename T1, typename T2, typename T4, typename T3>
double InnerProduct(const T1& Q, const T2& x, const T3& y, T4* workspace) {
  if (Q.rows() > 0) {
    workspace->noalias() = Q * y;
    return x.col(0).dot(workspace->col(0));
  } else {
    return x.col(0).dot(y.col(0));
  }
}

template <typename T1, typename T2, typename T3>
void QuadraticRepresentation(double x1_norm_squared,
                             double inner_product_of_x1_and_y1, double x0,
                             const T1& x1, double y0, const T2& y1,
                             double* z_q0, T3* z_q1) {
  // We use the formula from Example 11.12 of "Formally Real Jordan Algebras
  // and Their Applications to Optimization"  by Alizadeh, which states the
  // quadratic representation of x equals the linear map
  //                          2xx' - (det x) * R
  // where R is the reflection operator R = diag(1, -1, ..., -1) and det x is
  // the determinate of x = (x0, x1), i.e., det x = x0^2 - |x1|^2.
  double det_x = x0 * x0 - x1_norm_squared;
  double scale = 2 * (x0 * y0 + inner_product_of_x1_and_y1);
  *z_q0 = scale * x0 - det_x * y0;
  z_q1->noalias() = scale * x1 + det_x * y1;
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

template <typename T>
void SchurComplement(const Eigen::VectorXd& A0, const Eigen::MatrixXd& A_gram,
                     const double W0, double det_w,
                     const Eigen::MatrixXd& A_dot_w, bool initialize, T* G) {
  if (initialize) {
    G->noalias() = A0 * A0.transpose();
    G->noalias() -= A_gram;
    G->array() *= -det_w;
  } else {
    G->noalias() += det_w * (A_gram - A0 * A0.transpose());
  }
  G->noalias() += (A_dot_w + A0 * W0) * (A_dot_w + A0 * W0).transpose();
  G->noalias() += (A_dot_w + A0 * W0) * (A_dot_w + A0 * W0).transpose();
}

}  // namespace

DenseMatrix QuadraticConstraintBase::EvalAtQX(const DenseMatrix& X,
                                              DenseMatrix* QX) {
  if (Q_.rows() > 0) {
    QX->noalias() = Q_ * X;
    return A1_.transpose() * (*QX);
  } else {
    return A1_.transpose() * X;
  }
}

DenseMatrix QuadraticConstraintBase::EvalAtQX(const DenseMatrix& X, Ref* QX) {
  if (Q_.rows() > 0) {
    QX->noalias() = Q_ * X;
    return A1_.transpose() * (*QX);
  } else {
    return A1_.transpose() * X;
  }
}

void QuadraticConstraintBase::ComputeNegativeSlack(double inv_sqrt_mu,
                                                   const Ref& y,
                                                   double* minus_s_0,
                                                   Ref* minus_s_1) {
  *minus_s_0 = A0_.dot(y.col(0));
  *minus_s_0 -= C0_ * inv_sqrt_mu;
  minus_s_1->noalias() = A1_ * y;
  minus_s_1->noalias() -= C1_ * inv_sqrt_mu;
}

// Combine this with PrepareStep
void GetMuSelectionParameters(QuadraticConstraintBase* o, const Ref& y,
                              MuSelectionParameters* p) {
  Eigen::internal::set_is_malloc_allowed(false);
  auto* workspace = &o->workspace_;
  auto& minus_s_1 = workspace->temp1_1;
  double minus_s_0;
  o->ComputeNegativeSlack(1, y, &minus_s_0, &minus_s_1);

  double wsqrt_q0 = *o->workspace_.W0;
  auto& wsqrt_q1 = o->workspace_.temp3_1;
  wsqrt_q1 = o->workspace_.W1;

  double temp = Norm(o->Q_, wsqrt_q1, &workspace->temp2_1);
  Sqrt(temp, &wsqrt_q0, &wsqrt_q1);

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
  Eigen::internal::set_is_malloc_allowed(true);
}

void PrepareStep(QuadraticConstraintBase* o, const StepOptions& opt,
                 const Ref& y, StepInfo* info) {
  Eigen::internal::set_is_malloc_allowed(false);
  // d =  e - Q(w^{1/2})(C-A^y)
  double& wsqrt_q0 = *o->workspace_.W0;
  auto& wsqrt_q1 = o->workspace_.temp3_1;
  auto& d_q1 = o->workspace_.temp2_1;
  double& d_q0 = o->workspace_.d0;
  double& wsqrt_q1_norm_sqr = o->workspace_.wsqrt_q1_norm_sqr;

  {  // Use temp_1
    auto& minus_s_1 = o->workspace_.temp1_1;
    double minus_s_0;
    o->ComputeNegativeSlack(opt.inv_sqrt_mu, y, &minus_s_0, &minus_s_1);

    wsqrt_q1 = o->workspace_.W1;
    Sqrt(Norm(o->Q_, wsqrt_q1, &o->workspace_.temp2_1), &wsqrt_q0, &wsqrt_q1);

    wsqrt_q1_norm_sqr = SquaredNorm(o->Q_, wsqrt_q1, &o->workspace_.temp2_1);

    QuadraticRepresentation(
        wsqrt_q1_norm_sqr,
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

  Eigen::internal::set_is_malloc_allowed(true);
}

void QuadraticConstraintBase::Initialize() {
  DenseMatrix W;
  A_gram_ = EvalAtQX(A1_, &W);
}

bool TakeStep(QuadraticConstraintBase* o, const StepOptions& options) {
  auto& d_q1 = o->workspace_.temp2_1;
  double& d_q0 = o->workspace_.d0;
  double& wsqrt_q0 = *o->workspace_.W0;
  auto& wsqrt_q1 = o->workspace_.temp3_1;
  double& wsqrt_q1_norm_sqr = o->workspace_.wsqrt_q1_norm_sqr;
  if (options.step_size != 1) {
    d_q0 = options.step_size * d_q0;
    d_q1 = options.step_size * d_q1;
  }

  Exp(Norm(o->Q_, d_q1, &o->workspace_.temp1_1), &d_q0, &d_q1);
  const auto& expd_q1 = d_q1;
  const auto& expd_q0 = d_q0;

  QuadraticRepresentation(
      wsqrt_q1_norm_sqr,
      InnerProduct(o->Q_, wsqrt_q1, expd_q1, &o->workspace_.temp1_1), wsqrt_q0,
      wsqrt_q1, expd_q0, expd_q1, o->workspace_.W0, &o->workspace_.W1);
  return true;
}

void ConstructSchurComplementSystem(QuadraticConstraintBase* o, bool initialize,
                                    SchurComplementSystem* sys) {
  const auto& A0 = o->A0_;
  const auto& C0 = o->C0_;
  const auto& C1 = o->C1_;
  const auto& A_gram = o->A_gram_;
  auto& temp = o->workspace_.temp1_1;
  auto& A_dot_x = o->A_dot_x_;

  A_dot_x = o->EvalAtQX(o->workspace_.W1, &temp);

  auto& Q_W1 = o->workspace_.temp2_1;
  double det_w = (*o->workspace_.W0) * (*o->workspace_.W0) -
                 SquaredNorm(o->Q_, o->workspace_.W1, &Q_W1);

  if (initialize) {
    SchurComplement(A0, A_gram, *o->workspace_.W0, det_w, A_dot_x, true,
                    &sys->G);
    sys->AW.noalias() = A_dot_x + A0 * (*o->workspace_.W0);
    sys->AQc.noalias() = det_w * (o->EvalAtQX(C1, &temp) - A0 * C0);
  } else {
    SchurComplement(A0, A_gram, *o->workspace_.W0, det_w, A_dot_x, false,
                    &sys->G);
    sys->AW.noalias() += A_dot_x + A0 * (*o->workspace_.W0);
    sys->AQc.noalias() += det_w * (o->EvalAtQX(C1, &temp) - A0 * C0);
  }

  double scale;
  if (o->Q_.size() > 0) {
    scale = Q_W1.col(0).dot(C1.col(0)) + C0 * (*o->workspace_.W0);
  } else {
    scale = o->workspace_.W1.col(0).dot(C1.col(0)) + C0 * (*o->workspace_.W0);
  }
  sys->AQc.noalias() += 2 * (A_dot_x + A0 * (*o->workspace_.W0)) * scale;
}

}  // namespace conex
