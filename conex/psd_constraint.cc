#include "psd_constraint.h"

#include <cmath>
#include "eigen_decomp.h"

#include <unsupported/Eigen/MatrixFunctions>

using conex::jordan_algebra::NormInf;

// Applies update  W^{1/2}( exp ( e + W^{1/2} S W^{1/2} ) W^{1/2}
void PsdConstraint::GeodesicUpdate(double scale, const StepOptions& opt, Ref* SW) {
  auto& workspace = workspace_;
  auto& W = workspace.W;
  auto& expSW = workspace.temp_2;

  SW->diagonal().array() += opt.e_weight;
  if (scale != 1.0) {
    (*SW) *= scale;
  }
  expSW = SW->exp();
  W = W * expSW;
  *SW = W.transpose();
  W = (W + (*SW)) * 0.5;
}

// Applies update  W = W + W^{1/2} ( w_e +  W^{1/2} S W^{1/2} ) W^{1/2}
// which is a linearization of  W^{1/2}( exp ( W^{1/2} S W^{1/2} ) W^{1/2}
// We assume that SW = Ay.
void PsdConstraint::AffineUpdate(double w_e, Ref* SW) {
  auto& W = workspace_.W;
  auto& WSW = workspace_.temp_2;
  WSW = W * (*SW);
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
  auto& SW = workspace.temp_1;
  auto& SWSW = workspace.temp_2;

  if (opt.affine) {
    o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);
    SW = minus_s*W;
    o->AffineUpdate(opt.e_weight, &SW);
    return;
  }

  o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);
  SW = minus_s*W;

  int n = Rank(*o);
  double norminf = NormInf(SW + opt.e_weight*Eigen::MatrixXd::Identity(n, n));

  SWSW = SW*SW;
  double norm2 = SWSW.trace() + 2*SW.trace() + Rank(*o);
  double scale = 1;
  if (norminf * norminf > 2.0) {
    scale = 2.0/(norminf * norminf);
  }
  // if (norminf > 2.0) {
  //   scale = 2.0/(norminf);
  // }


  info->norminfd = norminf;
  info->normsqrd = norm2;
  o->GeodesicUpdate(scale, opt, &SW);
}

void SetIdentity(PsdConstraint* o) {
  o->workspace_.W.setZero();
  o->workspace_.W.diagonal().setConstant(1);
}

/*
    for (int i = 0; i < 10; i++) {
      double norminf = max(std::fabs(gw_eig.first * inv_sqrt_mu - 1),
                           std::fabs(gw_eig.second * inv_sqrt_mu - 1));
      if (norminf <= .9) {
        DUMP(norminf);
        break;
      } else {
        double inv_sqrt_mu_p = 1.9 / gw_eig.first;
        double inv_sqrt_mu_d = .1 / gw_eig.second;
        if (inv_sqrt_mu_p < inv_sqrt_mu_d) {
          inv_sqrt_mu = inv_sqrt_mu_p;
        } else {
          inv_sqrt_mu = inv_sqrt_mu_d;
        }
      }
    }
*/

// max(a x - 1, b y - 1)
// Goal: rescale max(a x - 1, b y - 1) = k.
//
//     max(a/k x - 1, b/y - 1)
void GetMuSelectionParameters(PsdConstraint* o,  const Ref& y, MuSelectionParameters* p) {
  using conex::jordan_algebra::SpectralRadius;
  auto* workspace = &o->workspace_;
  auto& minus_s = workspace->temp_1;
  auto& WsWs = workspace->temp_1;
  auto& Ws = workspace->temp_2;
  o->ComputeNegativeSlack(1, y, &minus_s);

  Ws.noalias() = workspace->W * minus_s;

  const auto gw_eig = SpectralRadius(Ws);
  const double lamda_max = -gw_eig.second;
  const double lamda_min = -gw_eig.first;

  if (p->gw_lambda_max < lamda_max) {
    p->gw_lambda_max = lamda_max;
  }
  if (p->gw_lambda_min > lamda_min) {
    p->gw_lambda_min = lamda_min;
  }
  WsWs = Ws*Ws;
  p->gw_norm_squared += WsWs.trace();
  p->gw_trace += -Ws.trace();
}






