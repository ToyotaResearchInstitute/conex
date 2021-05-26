#include <cmath>

#include "conex/approximate_eigenvalues.h"
#include "conex/exponential_map.h"
#include "conex/exponential_map_pade.h"
#include "conex/psd_constraint.h"

namespace conex {

using Eigen::VectorXd;

// Applies update  W^{1/2}( exp ( e + W^{1/2} S W^{1/2} ) W^{1/2}
void PsdConstraint::GeodesicUpdate(double scale, const StepOptions& opt,
                                   Ref* WS) {
  auto& workspace = workspace_;
  auto& W = workspace.W;
  auto& expWS = workspace.temp_2;

  WS->diagonal().array() += opt.e_weight;
  if (scale != 1.0) {
    (*WS) *= scale;
  }

  ExponentialMapPadeApproximation(*WS, &expWS);
  W = expWS * W;
  *WS = W.transpose();
  W = (W + (*WS)) * 0.5;
}

// Applies update  W = W + W^{1/2} ( w_e +  W^{1/2} S W^{1/2} ) W^{1/2}
// which is a linearization of  W^{1/2}( exp ( W^{1/2} S W^{1/2} ) W^{1/2}
// We assume that SW = Ay.
void PsdConstraint::AffineUpdate(double w_e, Ref* WS) {
  auto& W = workspace_.W;
  auto& WSW = workspace_.temp_2;
  WSW = (*WS) * W;
  if (w_e == 0) {
    W += WSW;
  } else {
    W.array() *= (1 + w_e);
    W += WSW;
  }
}

void PrepareStep(PsdConstraint* o, const StepOptions& opt, const Ref& y,
                 StepInfo* info) {
  auto& workspace = o->workspace_;
  auto& minus_s = workspace.temp_1;
  auto& W = workspace.W;
  auto& WS = workspace.temp_1;
  auto& WSWS = workspace.temp_2;

  if (opt.affine) {
    o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);
    WS = W * minus_s;
    o->AffineUpdate(opt.e_weight, &WS);
    return;
  }

  o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);
  WS = W * minus_s;

  int n = Rank(*o);
  // Use heuristic initialization of ApproximateEigenvalues.
  // Finds eigenvalues of -Q(w/2) s
  int index = 0;
  WS.diagonal().maxCoeff(&index);
  auto gw_eig =
      ApproximateEigenvalues(WS, workspace.W, minus_s.col(index), n / 2, true);

  // Get eigenvalues of e - Q(w/2) s.
  const double lambda_1 = std::fabs(opt.e_weight + gw_eig.minCoeff());
  const double lambda_2 = std::fabs(opt.e_weight + gw_eig.maxCoeff());
  double norminf = lambda_1;
  if (norminf < lambda_2) {
    norminf = lambda_2;
  }

  WSWS = WS * WS;
  double norm2 = WSWS.trace() + 2 * WS.trace() + Rank(*o);

  info->norminfd = norminf;
  info->normsqrd = norm2;
}

bool TakeStep(PsdConstraint* o, const StepOptions& options) {
  auto& WS = o->workspace_.temp_1;
  o->GeodesicUpdate(options.step_size, options, &WS);
  return true;
}

void SetIdentity(PsdConstraint* o) {
  o->workspace_.W.setZero();
  o->workspace_.W.diagonal().setConstant(1);
}

void GetWeightedSlackEigenvalues(PsdConstraint* o, const Ref& y,
                                 WeightedSlackEigenvalues* p) {
  auto* workspace = &o->workspace_;
  auto& minus_s = workspace->temp_1;
  auto& WSWS = workspace->temp_1;
  auto& WS = workspace->temp_2;
  o->ComputeNegativeSlack(1, y, &minus_s);

  WS.noalias() = workspace->W * minus_s;

#if 1
  int n = Rank(*o);
  // VectorXd r = minus_s.col(0);
  int index = 0;
  WS.diagonal().maxCoeff(&index);
  auto gw_eig =
      ApproximateEigenvalues(WS, workspace->W, minus_s.col(index), n / 2, true);
  // SWS
  const double lamda_max = -gw_eig.minCoeff();
  const double lamda_min = -gw_eig.maxCoeff();
#else
  const auto gw_eig = SpectrumBounds(WS);
  const double lamda_max = -gw_eig.second;
  const double lamda_min = -gw_eig.first;
#endif

  p->lambda_max = lamda_max;
  p->lambda_min = lamda_min;
  WSWS = WS * WS;
  p->frobenius_norm_squared = WSWS.trace();
  p->trace = -WS.trace();
}

}  // namespace conex
