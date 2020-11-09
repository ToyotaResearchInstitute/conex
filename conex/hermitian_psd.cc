#include "conex/hermitian_psd.h"
using Eigen::MatrixXd;

template<typename T>
void TakeStep(HermitianPsdConstraint<T>* o, const StepOptions& opt, const Ref& y, StepInfo* info) {
  typename T::Matrix minus_s;
  o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);

  // TODO: fix this approximation.
  double norminf = NormInfWeighted<T>(o->W, minus_s) - opt.e_weight;

  info->norminfd = norminf;
  info->normsqrd = T::TraceInnerProduct(T::QuadraticRepresentation(o->W, minus_s), minus_s) +
                   2 * T::TraceInnerProduct(o->W, minus_s) + Rank(*o);

  double scale = 1;
  if (norminf * norminf > 2.0) {
    scale = 2.0/(norminf * norminf);
    minus_s = T::ScalarMultiply(minus_s, scale);
  }

  auto exp_sw = o->GeodesicUpdate(o->W, minus_s);
  o->W = T::ScalarMultiply(exp_sw, std::exp(opt.e_weight * scale));
}

template void TakeStep(HermitianPsdConstraint<Real>* o, const StepOptions& opt, const Ref& y, StepInfo* info);
template void TakeStep(HermitianPsdConstraint<Complex>* o, const StepOptions& opt, const Ref& y, StepInfo* info);
template void TakeStep(HermitianPsdConstraint<Quaternions>* o, const StepOptions& opt, const Ref& y, StepInfo* info);
template void TakeStep(HermitianPsdConstraint<Octonions>* o, const StepOptions& opt, const Ref& y, StepInfo* info);
