#include "conex/hermitian_psd.h"
#include "conex/exponential_map.h"
#include "conex/error_checking_macros.h"
using Eigen::MatrixXd;

template<typename T>
void TakeStep(HermitianPsdConstraint<T>* o, const StepOptions& opt, const Ref& y, StepInfo* info) {
  typename T::Matrix minus_s;
  o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);

  auto WS = T::Multiply(o->W, minus_s);
  int n = Rank(*o);
  auto gw_eig = T::ApproximateEigenvalues(WS, o->W,  T::Random(n, 1), n / 2);
  const double lambda_1 = std::fabs(opt.e_weight+gw_eig.minCoeff());
  const double lambda_2 = std::fabs(opt.e_weight+gw_eig.maxCoeff());
  double norminf = lambda_1;
  if (norminf < lambda_2) {
    norminf = lambda_2; 
  }

  auto WSWS = T::Multiply(WS, WS);

  info->norminfd = norminf;
  info->normsqrd = WSWS.at(0).trace() + 2 * WS.at(0).trace() + Rank(*o);

  double scale = 1;
  if (norminf * norminf > 2.0) {
    scale = 2.0/(norminf * norminf);
    minus_s = T::ScalarMultiply(minus_s, scale);
  }

  WS.at(0).diagonal().array() += opt.e_weight;
  if (scale != 1.0) {
    WS = T::ScalarMultiply(WS, scale);
  }

  auto expWS = T::Zero(n, n);
  ExponentialMap(WS, &expWS);
  o->W = T::Multiply(expWS, o->W);
  o->W = T::ScalarMultiply(T::Add(o->W, T::ConjugateTranspose(o->W)), .5);
}

template<typename T>
void GetMuSelectionParameters(HermitianPsdConstraint<T>* o,  const Ref& y, MuSelectionParameters* p) {
  typename T::Matrix minus_s;
  o->ComputeNegativeSlack(1, y, &minus_s);

  int n = Rank(*o);
  auto WS = T::Multiply(o->W, minus_s);
  auto gw_eig = T::ApproximateEigenvalues(WS, o->W,  T::Random(n, 1), n / 2 + 1);

  const double lamda_max = -gw_eig.minCoeff();
  const double lamda_min = -gw_eig.maxCoeff();

  if (p->gw_lambda_max < lamda_max) {
    p->gw_lambda_max = lamda_max;
  }
  if (p->gw_lambda_min > lamda_min) {
    p->gw_lambda_min = lamda_min;
  }
  if (p->gw_lambda_max < lamda_max) {
    p->gw_lambda_max = lamda_max;
  }
  if (p->gw_lambda_min > lamda_min) {
    p->gw_lambda_min = lamda_min;
  }
  auto WSWS = T::Multiply(WS, WS);
  p->gw_norm_squared += WSWS.at(0).trace();
  p->gw_trace += -WS.at(0).trace();
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
    
    typename T::Matrix AW;
    typename T::Matrix WAW;
    if (initialize) {
      for (int i = 0; i < m; i++) {
        if constexpr(std::is_same<T, Octonions>::value) {
          WAW = T::QuadraticRepresentation(W, o->constraint_matrices_.at(i));
        } else {
          AW = T::Multiply(o->constraint_matrices_.at(i), W);
          WAW = T::Multiply(W, AW);
        }
        for (int j = i; j < m; j++) {
          (*G)(j, i) = o->EvalDualConstraint(j, WAW);
        }
        if constexpr(std::is_same<T, Octonions>::value) {
          sys->AW(i, 0) = o->EvalDualConstraint(i, W);
        } else {
          sys->AW(i, 0) = AW.at(0).trace(); 
        }
        sys->AQc(i, 0) = o->EvalDualObjective(WAW);
      }
    } else {
      for (int i = 0; i < m; i++) {
        if constexpr(std::is_same<T, Octonions>::value) {
          WAW = T::QuadraticRepresentation(W, o->constraint_matrices_.at(i));
        } else {
          AW = T::Multiply(o->constraint_matrices_.at(i), W);
          WAW = T::Multiply(W, AW);
        }

        for (int j = i; j < m; j++) {
          (*G)(j, i) += o->EvalDualConstraint(j, WAW);
        }

        if constexpr(std::is_same<T, Octonions>::value) {
          sys->AW(i, 0) += o->EvalDualConstraint(i, W);
        } else {
          sys->AW(i, 0) += AW.at(0).trace(); 
        }
        sys->AQc(i, 0) += o->EvalDualObjective(WAW);
      }
    }
}




template void ConstructSchurComplementSystem(HermitianPsdConstraint<Real>* o, bool initialize, SchurComplementSystem* sys);

template void ConstructSchurComplementSystem(HermitianPsdConstraint<Complex>* o, bool initialize, SchurComplementSystem* sys);

template void ConstructSchurComplementSystem(HermitianPsdConstraint<Quaternions>* o, bool initialize, SchurComplementSystem* sys);

template void ConstructSchurComplementSystem(HermitianPsdConstraint<Octonions>* o, bool initialize, SchurComplementSystem* sys);



template<typename H>
bool UpdateLinearOperator(HermitianPsdConstraint<H>* o,  
                          double val, int var, int r, int c, int dim)  {

  CONEX_DEMAND(dim < H::HyperComplexDimension(), "Complex dimension out of bounds.");
  CONEX_DEMAND(r < o->rank_ && c < o->rank_, "Matrix dimension out of bounds.");

  using T = HermitianPsdConstraint<H>;
  if constexpr(std::is_same<T, Octonions>::value) {
    if (dim >= 3)  {
      return false;
    }
  }

  for (int i = o->constraint_matrices_.size(); i <= var; i++) {
    o->constraint_matrices_.push_back(H::Zero(o->rank_, o->rank_));
  }
  o->constraint_matrices_.at(var).at(dim)(r, c) = val;
  if (dim == 0) {
    o->constraint_matrices_.at(var).at(dim)(c, r) = val;
  } else {
    o->constraint_matrices_.at(var).at(dim)(c, r) = -val;
  }
  return CONEX_SUCCESS;
}

template bool UpdateLinearOperator(HermitianPsdConstraint<Complex>* o,  double val,
int var, int r, int c, int dim);
template bool UpdateLinearOperator(HermitianPsdConstraint<Real>* o,  double val,
int var, int r, int c, int dim);
template bool UpdateLinearOperator(HermitianPsdConstraint<Quaternions>* o,  double val,
int var, int r, int c, int dim);
template bool UpdateLinearOperator(HermitianPsdConstraint<Octonions>* o,  double val,
int var, int r, int c, int dim);


template<typename H>
bool UpdateAffineTerm(HermitianPsdConstraint<H>* o,  
                          double val,  int r, int c, int dim)  {
  CONEX_DEMAND(dim < H::HyperComplexDimension(), "Complex dimension out of bounds.");
  CONEX_DEMAND(r < o->rank_ && c < o->rank_, "Matrix dimension out of bounds.");

  using T = HermitianPsdConstraint<H>;
  if constexpr(std::is_same<T, Octonions>::value) {
    if (dim >= 3)  {
      return false;
    }
  }

  if (o->constraint_affine_.size() == 0) {
    o->constraint_affine_ = H::Zero(o->rank_, o->rank_);
  }
  
  o->constraint_affine_.at(dim)(r, c) = val;
  if (dim == 0) {
    o->constraint_affine_.at(dim)(c, r) = val;
  } else {
    o->constraint_affine_.at(dim)(c, r) = -val;
  }
 
  return CONEX_SUCCESS;
}

template bool UpdateAffineTerm(HermitianPsdConstraint<Complex>* o,  double val,
int r, int c, int dim);
template bool UpdateAffineTerm(HermitianPsdConstraint<Real>* o,  double val,
int r, int c, int dim);
template bool UpdateAffineTerm(HermitianPsdConstraint<Quaternions>* o,  double val,
int r, int c, int dim);
template bool UpdateAffineTerm(HermitianPsdConstraint<Octonions>* o,  double val,
int r, int c, int dim);




