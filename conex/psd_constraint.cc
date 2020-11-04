#include <cmath>

#include "conex/psd_constraint.h"
#include "conex/approximate_eigenvalues.h"
#include "conex/matrix_exponential.h"

using conex::jordan_algebra::SpectralRadius;
using conex::jordan_algebra::SpectrumBounds;
using Eigen::VectorXd;

// Applies update  W^{1/2}( exp ( e + W^{1/2} S W^{1/2} ) W^{1/2}
void PsdConstraint::GeodesicUpdate(double scale, const StepOptions& opt, Ref* WS) {
  auto& workspace = workspace_;
  auto& W = workspace.W;
  auto& expWS = workspace.temp_2;

  WS->diagonal().array() += opt.e_weight;
  if (scale != 1.0) {
    (*WS) *= scale;
  }

  MatrixExponential(*WS, &expWS);
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

void TakeStep(PsdConstraint* o, const StepOptions& opt, const Ref& y, StepInfo* info) {
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
  // The spectral radius of |WS + kI| is the inf-norm of W^{1/2} S W^{1/2} + kI
  // given that they have the same eigenvalues.
#if 0
  double norminf = SpectralRadius(WS + opt.e_weight*Eigen::MatrixXd::Identity(n, n));
#else
  // Heuristic.
  int index = 0; WS.diagonal().maxCoeff(&index);
  auto gw_eig = ApproximateEigenvalues(WS, workspace.W,  minus_s.col(index), n / 2, true);
  const double lambda_1 = std::fabs(opt.e_weight+gw_eig.minCoeff());
  const double lambda_2 = std::fabs(opt.e_weight+gw_eig.maxCoeff());
  double norminf = lambda_1;
  if (norminf < lambda_2) {
    norminf = lambda_2; 
  }
#endif



  WSWS = WS*WS;
  double norm2 = WSWS.trace() + 2*WS.trace() + Rank(*o);
  double scale = 1;
  if (norminf * norminf > 2.0) {
    scale = 2.0/(norminf * norminf);
  }

  info->norminfd = norminf;
  info->normsqrd = norm2;
  o->GeodesicUpdate(scale, opt, &WS);
}

void SetIdentity(PsdConstraint* o) {
  o->workspace_.W.setZero();
  o->workspace_.W.diagonal().setConstant(1);
}

void GetMuSelectionParameters(PsdConstraint* o,  const Ref& y, MuSelectionParameters* p) {
  using conex::jordan_algebra::SpectralRadius;
  auto* workspace = &o->workspace_;
  auto& minus_s = workspace->temp_1;
  auto& WSWS = workspace->temp_1;
  auto& WS = workspace->temp_2;
  o->ComputeNegativeSlack(1, y, &minus_s);

  WS.noalias() =  workspace->W * minus_s;

#if 1
  int n = Rank(*o);
  //VectorXd r = minus_s.col(0); 
  int index = 0; WS.diagonal().maxCoeff(&index);
  auto gw_eig = ApproximateEigenvalues(WS, workspace->W,  minus_s.col(index), n / 2, true);
  // SWS
  const double lamda_max = -gw_eig.minCoeff();
  const double lamda_min = -gw_eig.maxCoeff();
#else
  const auto gw_eig = SpectrumBounds(WS);
  const double lamda_max = -gw_eig.second;
  const double lamda_min = -gw_eig.first;
#endif

  if (p->gw_lambda_max < lamda_max) {
    p->gw_lambda_max = lamda_max;
  }
  if (p->gw_lambda_min > lamda_min) {
    p->gw_lambda_min = lamda_min;
  }
  WSWS = WS*WS;
  p->gw_norm_squared += WSWS.trace();
  p->gw_trace += -WS.trace();
}
