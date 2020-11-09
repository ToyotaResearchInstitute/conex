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

template<typename T>
void GetMuSelectionParameters(HermitianPsdConstraint<T>* o,  const Ref& y, MuSelectionParameters* p) {
  typename T::Matrix minus_s;
  o->ComputeNegativeSlack(1, y, &minus_s);
 
  p->gw_lambda_max = NormInfWeighted<T>(o->W, minus_s);

  // <e, Q(w^{1/2}) s)
  p->gw_trace -= T::TraceInnerProduct(o->W, minus_s);
  p->gw_norm_squared += T::TraceInnerProduct(T::QuadraticRepresentation(o->W, minus_s), minus_s);
}

template void TakeStep(HermitianPsdConstraint<Real>* o, const StepOptions& opt, const Ref& y, StepInfo* info);
template void TakeStep(HermitianPsdConstraint<Complex>* o, const StepOptions& opt, const Ref& y, StepInfo* info);
template void TakeStep(HermitianPsdConstraint<Quaternions>* o, const StepOptions& opt, const Ref& y, StepInfo* info);


template void GetMuSelectionParameters(HermitianPsdConstraint<Real>* o,  const Ref& y, MuSelectionParameters* p);

template void GetMuSelectionParameters(HermitianPsdConstraint<Complex>* o,  const Ref& y, MuSelectionParameters* p);

template void GetMuSelectionParameters(HermitianPsdConstraint<Quaternions>* o,  const Ref& y, MuSelectionParameters* p);


template<>
void TakeStep(HermitianPsdConstraint<Octonions>* o, const StepOptions& opt, const Ref& y, StepInfo* info) {
  using T = Octonions;
  typename T::Matrix minus_s;
  o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);

  // TODO: fix this approximation.

  // || e - Q(w^{1/2}) s\|
  double trace_ws = T::TraceInnerProduct(o->W, minus_s);
  info->normsqrd = T::TraceInnerProduct(T::QuadraticRepresentation(o->W, minus_s), minus_s) +
                   2 * trace_ws + Rank(*o);

  // TODO(FrankPermenter): replace this heuristic approximation. 
  info->norminfd = std::fabs(info->normsqrd)/(1e-15 + std::fabs(trace_ws + Rank(*o)));

  // TODO(FrankPermenter): Update this to infinity norm or to a better upper bound.
  double scale = 1;
  if (info->normsqrd > 2.0) {
    scale = 2.0/(info->normsqrd);
    minus_s = T::ScalarMultiply(minus_s, scale);
  }

  auto exp_sw = o->GeodesicUpdate(o->W, minus_s);
  o->W = T::ScalarMultiply(exp_sw, std::exp(opt.e_weight * scale));
}


template<>
void GetMuSelectionParameters(HermitianPsdConstraint<Octonions>* o,  const Ref& y, MuSelectionParameters* p) {
  using T = Octonions;
  typename T::Matrix minus_s;
  o->ComputeNegativeSlack(1, y, &minus_s);

  double normsqrd = T::TraceInnerProduct(T::QuadraticRepresentation(o->W, minus_s), minus_s);

  // Heuristic approximation based off of inequality:  |x|_1 |x|_{\infty} >= |x|^2_2.
  p->gw_lambda_max =  std::fabs(normsqrd)/(1e-15 + std::fabs(T::TraceInnerProduct(o->W, minus_s)));

  p->gw_trace -= T::TraceInnerProduct(o->W, minus_s);
  p->gw_norm_squared += T::TraceInnerProduct(T::QuadraticRepresentation(o->W, minus_s), minus_s);
}



template<typename T>
void ConstructSchurComplementSystem(HermitianPsdConstraint<T>* o, bool initialize, SchurComplementSystem* sys) {
    auto G = &sys->G;
    auto& W = o->W; 
    int m = o->constraint_matrices_.size();
    if (initialize) {
      for (int i = 0; i < m; i++) {
        // 
        typename T::Matrix QA = T::QuadraticRepresentation(W, o->constraint_matrices_.at(i));
        for (int j = i; j < m; j++) {
          (*G)(j, i) = o->EvalDualConstraint(j, QA);
        }

        sys->AW(i, 0)  = o->EvalDualConstraint(i, W);
        sys->AQc(i, 0) = o->EvalDualObjective(QA);
      }
    } else {
      for (int i = 0; i < m; i++) {
        typename T::Matrix QA = T::QuadraticRepresentation(W, o->constraint_matrices_.at(i));
        for (int j = i; j < m; j++) {
          (*G)(j, i) += o->EvalDualConstraint(j, QA);
        }

        sys->AW(i, 0)  += o->EvalDualConstraint(i, W);
        sys->AQc(i, 0) += o->EvalDualObjective(QA);
      }
    }
  }

template void ConstructSchurComplementSystem(HermitianPsdConstraint<Real>* o, bool initialize, SchurComplementSystem* sys);

template void ConstructSchurComplementSystem(HermitianPsdConstraint<Complex>* o, bool initialize, SchurComplementSystem* sys);

template void ConstructSchurComplementSystem(HermitianPsdConstraint<Quaternions>* o, bool initialize, SchurComplementSystem* sys);

template void ConstructSchurComplementSystem(HermitianPsdConstraint<Octonions>* o, bool initialize, SchurComplementSystem* sys);

