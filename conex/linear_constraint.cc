#define EIGEN_NO_MALLOC
#include "linear_constraint.h"
#include "newton_step.h"

namespace conex {

void SetIdentity(LinearConstraint* o) { o->workspace_.W.setConstant(1); }

// TODO: use e_weight and c_weight
void PrepareStep(LinearConstraint* o, const StepOptions& options, const Ref& y,
                 StepInfo* info) {
  auto* workspace = &o->workspace_;
  auto& minus_s = workspace->temp_1;
  if (!options.affine) {
    o->ComputeNegativeSlack(options.c_weight, y, &minus_s);
    auto& W = workspace->W;
    auto& SW = workspace->temp_1;
    auto& d = workspace->temp_2;
    SW = minus_s.cwiseProduct(W);

    int n = SW.rows();

    d = SW + DenseMatrix::Ones(n, 1);
    double norminf = (d).array().abs().maxCoeff();
    info->norminfd = norminf;
    info->normsqrd = d.squaredNorm();

  } else {
    o->ComputeNegativeSlack(0, y, &minus_s);
    TakeStep(o, options);
  }
}

bool TakeStep(LinearConstraint* o, const StepOptions& options) {
  if (!options.affine) {
    auto& d = o->workspace_.temp_2;
    auto& W = o->workspace_.W;
    if (options.step_size != 1) {
      d.array() *= options.step_size;
    }
    d = d.array().exp();
    W = W.cwiseProduct(d);
  } else {
    auto& minus_s = o->workspace_.temp_1;
    o->AffineUpdate(minus_s);
  }
  return true;
}

// Eigenvalues of Q(w/2)(C - A'y).
void GetWeightedSlackEigenvalues(LinearConstraint* o, const Ref& y,
                                 double c_weight, WeightedSlackEigenvalues* p) {
  auto* workspace = &o->workspace_;
  auto& minus_s = workspace->temp_1;
  auto& Ws = workspace->temp_2;
  o->ComputeNegativeSlack(c_weight, y, &minus_s);
  Ws.noalias() = workspace->W.cwiseProduct(minus_s);

  const double lamda_max = -Ws.minCoeff();
  const double lamda_min = -Ws.maxCoeff();

  p->lambda_max = lamda_max;
  p->lambda_min = lamda_min;
  p->frobenius_norm_squared = Ws.squaredNorm();
  p->trace = -Ws.sum();
}

void LinearConstraint::ComputeNegativeSlack(double inv_sqrt_mu, const Ref& y,
                                            Ref* minus_s) {
  minus_s->noalias() = (constraint_matrix_)*y.topRows(number_of_variables());
  minus_s->noalias() -= (constraint_affine_)*inv_sqrt_mu;
}

void LinearConstraint::AffineUpdate(const Ref& minus_s) {
  auto& W = workspace_.W;
  auto& SW = workspace_.temp_1;
  SW = minus_s.cwiseProduct(W);
  W += W.cwiseProduct(SW);
}

void ConstructSchurComplementSystem(LinearConstraint* o, bool initialize,
                                    SchurComplementSystem* sys) {
  const auto& W = o->workspace_.W;
  auto G = &sys->G;

  auto& WA = o->workspace_.weighted_constraints;
  auto& WC = o->workspace_.temp_1;
  int m = o->number_of_variables();

  WA = W.asDiagonal() * (o->constraint_matrix_);
  WC = W.cwiseProduct(o->constraint_affine_);

  if (initialize) {
    sys->inner_product_of_w_and_c = WC.sum();
    if (G->rows() != m) {
      G->setZero();
      sys->AW.setZero();
      sys->AQc.setZero();
    }
    (*G).topLeftCorner(m, m).noalias() = WA.transpose() * WA;
    sys->AW.topRows(m).noalias() = o->constraint_matrix_.transpose() * W;
    sys->AQc.topRows(m).noalias() = WA.transpose() * WC;
  } else {
    sys->inner_product_of_w_and_c += WC.sum();
    (*G).topLeftCorner(m, m).noalias() += WA.transpose() * WA;
    sys->AW.topRows(m).noalias() += o->constraint_matrix_.transpose() * W;
    sys->AQc.topRows(m).noalias() += WA.transpose() * WC;
  }
}

}  // namespace conex
