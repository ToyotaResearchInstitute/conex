#include "conex/hermitian_psd.h"
using Eigen::MatrixXd;

MatrixXd ToMat(const Real::Matrix& x) {
  MatrixXd y(3, 3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      y(i, j) = x.at(LinIndex(i, j))(0);
    }
  }
  return y;
}

template<typename T>
void DUMPA(const T& M) {
  for (const auto& e : M) {
    DUMP(e);
  }
}
void TakeStep(HermitianPsdConstraint<Real>* o, const StepOptions& opt, const Ref& y, StepInfo* info) {
  using T = Real;
  typename T::Matrix minus_s;
  o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);

  // TODO: fix this approximation.
  double norminf = NormInfWeighted<T>(o->W, minus_s) - opt.e_weight;

  double scale = 1;
  // if (norminf * norminf > 2.0) {
  //   scale = 2.0/(norminf * norminf);
  //   minus_s = T::ScalarMult(minus_s, scale);
  // }

  auto exp_sw = o->GeodesicUpdate(o->W, minus_s);
  o->W = T::ScalarMult(exp_sw, std::exp(opt.e_weight * scale));
  info->norminfd = norminf;
  info->normsqrd = -66;
}

void TakeStep(HermitianPsdConstraint<Octonions>* o, const StepOptions& opt, const Ref& y, StepInfo* info) {
  using T = Octonions;
  typename T::Matrix minus_s;
  o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);

  // TODO: fix this approximation.
  double norminf = NormInfWeighted<T>(o->W, minus_s) - opt.e_weight;

  double scale = 1;
  // if (norminf * norminf > 2.0) {
  //   scale = 2.0/(norminf * norminf);
  //   minus_s = T::ScalarMult(minus_s, scale);
  // }

  auto exp_sw = o->GeodesicUpdate(o->W, minus_s);
  o->W = T::ScalarMult(exp_sw, std::exp(opt.e_weight * scale));
  info->norminfd = norminf;
  info->normsqrd = -66;
}

void TakeStep(HermitianPsdConstraint<Complex>* o, const StepOptions& opt, const Ref& y, StepInfo* info) {
  using T = Complex;
  typename T::Matrix minus_s;
  o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);

  // TODO: fix this approximation.
  double norminf = NormInfWeighted<T>(o->W, minus_s) - opt.e_weight;

  double scale = 1;
  // if (norminf * norminf > 2.0) {
  //   scale = 2.0/(norminf * norminf);
  //   minus_s = T::ScalarMult(minus_s, scale);
  // }

  auto exp_sw = o->GeodesicUpdate(o->W, minus_s);
  o->W = T::ScalarMult(exp_sw, std::exp(opt.e_weight * scale));
  info->norminfd = norminf;
  info->normsqrd = -66;
}

void TakeStep(HermitianPsdConstraint<Quaternions>* o, const StepOptions& opt, const Ref& y, StepInfo* info) {
  using T = Quaternions;
  typename T::Matrix minus_s;
  o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);

  // TODO: fix this approximation.
  double norminf = NormInfWeighted<T>(o->W, minus_s) - opt.e_weight;

  double scale = 1;
  // if (norminf * norminf > 2.0) {
  //   scale = 2.0/(norminf * norminf);
  //   minus_s = T::ScalarMult(minus_s, scale);
  // }

  auto exp_sw = o->GeodesicUpdate(o->W, minus_s);
  o->W = T::ScalarMult(exp_sw, std::exp(opt.e_weight * scale));
  info->norminfd = norminf;
  info->normsqrd = -66;
}
