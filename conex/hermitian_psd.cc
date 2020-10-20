#include "conex/hermitian_psd.h"

template<typename T>
void DUMPA(const T& M) {
  for (const auto& e : M) {
    DUMP(e);
  }
}
void TakeStep(HermitianPsdConstraint<Real>* o, const StepOptions& opt, const Ref& y, StepInfo* info) {
  {
    using T = Real;
    typename T::Matrix minus_s;
    o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);

    int n = Rank(*o);
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
}
