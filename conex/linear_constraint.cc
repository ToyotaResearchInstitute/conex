#include "linear_constraint.h"
#include "newton_step.h"

namespace conex {
using Eigen::VectorXd;

bool FindMinimumMu(const VectorXd& d0, const VectorXd& delta, double dinfmax,
                   LineSearchOutput* output) {
  auto& upper_bound = output->upper_bound;
  auto& lower_bound = output->lower_bound;
  double upper_bound_i;
  double lower_bound_i;
  double temp;
  int i = 0;

  for (i = 0; i < d0.size(); i++) {
    upper_bound_i = (dinfmax - d0(i)) / delta(i);
    lower_bound_i = (-dinfmax - d0(i)) / delta(i);

    if (lower_bound_i > upper_bound_i) {
      temp = upper_bound_i;
      upper_bound_i = lower_bound_i;
      lower_bound_i = temp;
    }

    if (upper_bound_i < upper_bound || i == 0) {
      upper_bound = upper_bound_i;
    }

    if (lower_bound_i > lower_bound || i == 0) {
      lower_bound = lower_bound_i;
    }
  }

  bool success = true;
  if (lower_bound > upper_bound) {
    success = false;
  }
  return success;
}
// TODO(FrankPermenter): Remove dynamic memory allocation and define EIGEN_NO_MALLOC 
bool PerformLineSearch(LinearConstraint* o, const LineSearchParameters& params,
                       const Ref& y0, const Ref& y1, LineSearchOutput* output) {
  auto* workspace = &o->workspace_;

  auto& minus_s = workspace->temp_1;
  auto& W = workspace->W;
  auto& SW = workspace->temp_1;
  auto& d0 = workspace->temp_2;

  o->ComputeNegativeSlack(params.c0_weight, y0, &minus_s);
  SW = minus_s.cwiseProduct(W);
  int n = SW.rows();
  d0 = SW + DenseMatrix::Ones(n, 1);

  o->ComputeNegativeSlack(params.c1_weight, y1, &minus_s);
  SW = minus_s.cwiseProduct(W);
  VectorXd d1 = SW + DenseMatrix::Ones(n, 1);

  bool success = FindMinimumMu(d0, d1 - d0, params.dinf_upper_bound, output);
  return !success;
}

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
    sys->inner_product_of_c_and_Qc = WC.squaredNorm();
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
    sys->inner_product_of_c_and_Qc += WC.squaredNorm();
    (*G).topLeftCorner(m, m).noalias() += WA.transpose() * WA;
    sys->AW.topRows(m).noalias() += o->constraint_matrix_.transpose() * W;
    sys->AQc.topRows(m).noalias() += WA.transpose() * WC;
  }
}

bool UpdateLinearOperator(LinearConstraint* o, double val, int var, int r,
                          int c, int dim) {
  CONEX_DEMAND(dim == 0, "Complex linear constraints not supported.");
  CONEX_DEMAND(c == 0, "Linear constraint is not matrix valued.");
  CONEX_DEMAND(r < o->constraint_matrix_.rows(), "Row index out of bounds.");
  CONEX_DEMAND((var >= 0) && (r >= 0), "Indices cannot be negative.");

  o->constraint_matrix_(r, var) = val;
  return CONEX_SUCCESS;
}

bool UpdateAffineTerm(LinearConstraint* o, double val, int r, int c, int dim) {
  CONEX_DEMAND(dim == 0, "Complex linear cone not supported.");
  CONEX_DEMAND(c == 0, "Linear constraint is not matrix valued.");
  CONEX_DEMAND(r < o->constraint_matrix_.rows(), "Row index out of bounds.");
  CONEX_DEMAND(r >= 0, "Indices cannot be negative.");

  o->constraint_affine_(r) = val;
  return CONEX_SUCCESS;
}

}  // namespace conex
