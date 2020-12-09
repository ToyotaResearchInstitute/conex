#define EIGEN_NO_MALLOC
#include "linear_constraint.h"
#include "newton_step.h"

void SetIdentity(LinearConstraint* o) { o->workspace_.W.setConstant(1); }

// TODO: use e_weight and c_weight
void TakeStep(LinearConstraint* o, const StepOptions& opt, const Ref& y,
              StepInfo* info) {
  auto* workspace = &o->workspace_;
  auto& minus_s = workspace->temp_1;
  if (!opt.affine) {
    o->ComputeNegativeSlack(opt.inv_sqrt_mu, y, &minus_s);
  } else {
    o->ComputeNegativeSlack(0, y, &minus_s);
  }
  if (!opt.affine) {
    o->GeodesicUpdate(minus_s, info);
  } else {
    o->AffineUpdate(minus_s);
  }
}

void GetMuSelectionParameters(LinearConstraint* o, const Ref& y,
                              MuSelectionParameters* p) {
  auto* workspace = &o->workspace_;
  auto& minus_s = workspace->temp_1;
  auto& Ws = workspace->temp_2;
  o->ComputeNegativeSlack(1, y, &minus_s);
  Ws.noalias() = workspace->W.cwiseProduct(minus_s);

  const double lamda_max = -Ws.minCoeff();
  const double lamda_min = -Ws.maxCoeff();

  if (p->gw_lambda_max < lamda_max) {
    p->gw_lambda_max = lamda_max;
  }
  if (p->gw_lambda_min > lamda_min) {
    p->gw_lambda_min = lamda_min;
  }
  p->gw_norm_squared += Ws.squaredNorm();
  p->gw_trace += -Ws.sum();
}

void LinearConstraint::ComputeNegativeSlack(double inv_sqrt_mu, const Ref& y,
                                            Ref* minus_s) {
  minus_s->noalias() = (constraint_matrix_)*y.topRows(number_of_variables());
  minus_s->noalias() -= (constraint_affine_)*inv_sqrt_mu;
}

void LinearConstraint::GeodesicUpdate(const Ref& minus_s, StepInfo* info) {
  auto& W = workspace_.W;
  auto& SW = workspace_.temp_1;
  auto& d = workspace_.temp_2;
  auto& expSW = workspace_.temp_2;
  SW = minus_s.cwiseProduct(W);

  int n = SW.rows();

  d = SW + DenseMatrix::Ones(n, 1);
  double norminf = (d).array().abs().maxCoeff();

  info->norminfd = norminf;
  info->normsqrd = d.squaredNorm();

  double scale = 1;
  if (norminf * norminf > 2.0) {
    scale = 2.0 / (norminf * norminf);
    SW = SW * scale;
  }

  // double scale = 1;
  // if (norminf > 2.0) {
  //   scale = 2.0/(norminf);
  //   SW = SW*scale;
  // }

  expSW = SW.array().exp();
  W = W.cwiseProduct(expSW);
  W = std::exp(scale) * W;
}

void LinearConstraint::AffineUpdate(const Ref& minus_s) {
  auto& W = workspace_.W;
  auto& SW = workspace_.temp_1;
  SW = minus_s.cwiseProduct(W);
  W += W.cwiseProduct(SW);
}

// WA = W * A -> Weight all the rows.
// A' W W A   -> Compute dot product of columns.
// Want random access to columns.
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
    if (G->rows() != m) {
      G->setZero();
      sys->AW.setZero();
      sys->AQc.setZero();
    }
    (*G).topLeftCorner(m, m).noalias() = WA.transpose() * WA;
    sys->AW.topRows(m).noalias() = o->constraint_matrix_.transpose() * W;
    sys->AQc.topRows(m).noalias() = WA.transpose() * WC;
  } else {
    (*G).topLeftCorner(m, m).noalias() += WA.transpose() * WA;
    sys->AW.topRows(m).noalias() += o->constraint_matrix_.transpose() * W;
    sys->AQc.topRows(m).noalias() += WA.transpose() * WC;
  }
}
