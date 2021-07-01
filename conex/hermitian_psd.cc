#include "conex/hermitian_psd.h"
#include "conex/error_checking_macros.h"
#include "conex/exponential_map.h"

namespace conex {

using Eigen::MatrixXd;

template <typename T>
bool TakeStep(HermitianPsdConstraint<T>* o, const StepOptions& opt) {
  auto& WS = o->WS;
  WS.at(0).diagonal().array() += opt.e_weight;
  double scale = opt.step_size;
  if (scale != 1.0) {
    WS = T::ScalarMultiply(WS, scale);
  }

  int n = Rank(*o);
  auto expWS = T::Zero(n, n);
  ExponentialMap(WS, &expWS);
  o->W = T::Multiply(expWS, o->W);
  o->W = T::ScalarMultiply(T::Add(o->W, T::ConjugateTranspose(o->W)), .5);

  // TODO(FrankPermenter): Remove this hack, which provides
  // the dual-variable-interface access to real part of W.
  if (o->W.at(0).data() != o->workspace_.W.data()) {
    new (&o->workspace_.W)
        Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>(o->W.at(0).data(), n, n);
  }
  return true;
}

template <typename T>
void PrepareStep(HermitianPsdConstraint<T>* o, const StepOptions& opt,
                 const Ref& y, StepInfo* info) {
  auto& minus_s = o->minus_s;
  auto& WS = o->WS;
  o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);

  WS = T::Multiply(o->W, minus_s);
  int n = Rank(*o);
  auto gw_eig = T::ApproximateEigenvalues(WS, o->W, T::Random(n, 1), n / 2 + 1);
  const double lambda_1 = std::fabs(opt.e_weight + gw_eig.minCoeff());
  const double lambda_2 = std::fabs(opt.e_weight + gw_eig.maxCoeff());
  double norminf = lambda_1;
  if (norminf < lambda_2) {
    norminf = lambda_2;
  }

  auto WSWS = T::Multiply(WS, WS);

  info->norminfd = norminf;
  info->normsqrd = WSWS.at(0).trace() + 2 * WS.at(0).trace() + Rank(*o);
}

template <typename T>
void GetWeightedSlackEigenvalues(HermitianPsdConstraint<T>* o, const Ref& y,
                                 double c_weight, WeightedSlackEigenvalues* p) {
  typename T::Matrix minus_s;
  o->ComputeNegativeSlack(c_weight, y, &minus_s);

  int n = Rank(*o);
  auto WS = T::Multiply(o->W, minus_s);
  auto gw_eig = T::ApproximateEigenvalues(WS, o->W, T::Random(n, 1), n / 2 + 1);

  const double lambda_max = -gw_eig.minCoeff();
  const double lambda_min = -gw_eig.maxCoeff();

  p->lambda_max = lambda_max;
  p->lambda_min = lambda_min;
  auto WSWS = T::Multiply(WS, WS);
  p->frobenius_norm_squared = WSWS.at(0).trace();
  p->trace = -WS.at(0).trace();
}

template void PrepareStep(HermitianPsdConstraint<Real>* o,
                          const StepOptions& opt, const Ref& y, StepInfo* info);
template void PrepareStep(HermitianPsdConstraint<Complex>* o,
                          const StepOptions& opt, const Ref& y, StepInfo* info);
template void PrepareStep(HermitianPsdConstraint<Quaternions>* o,
                          const StepOptions& opt, const Ref& y, StepInfo* info);

template bool TakeStep(HermitianPsdConstraint<Real>* o, const StepOptions& opt);
template bool TakeStep(HermitianPsdConstraint<Complex>* o,
                       const StepOptions& opt);
template bool TakeStep(HermitianPsdConstraint<Quaternions>* o,
                       const StepOptions& opt);

template void GetWeightedSlackEigenvalues(HermitianPsdConstraint<Real>* o,
                                          const Ref& y, double c_weight,
                                          WeightedSlackEigenvalues* p);
template void GetWeightedSlackEigenvalues(HermitianPsdConstraint<Complex>* o,
                                          const Ref& y, double c_weight,
                                          WeightedSlackEigenvalues* p);
template void GetWeightedSlackEigenvalues(
    HermitianPsdConstraint<Quaternions>* o, const Ref& y, double c_weight,
    WeightedSlackEigenvalues* p);

template <>
bool TakeStep(HermitianPsdConstraint<Octonions>* o, const StepOptions& opt) {
  using T = Octonions;
  auto& minus_s = o->minus_s;
  double scale = opt.step_size;

  if (scale != 1) {
    minus_s = T::ScalarMultiply(minus_s, scale);
  }

  o->W = GeodesicUpdateScaled(o->W, minus_s);
  return true;
}

template <>
void PrepareStep(HermitianPsdConstraint<Octonions>* o, const StepOptions& opt,
                 const Ref& y, StepInfo* info) {
  using T = Octonions;
  auto& minus_s = o->minus_s;
  o->ComputeNegativeSlack(opt.c_weight, y, &minus_s);

  // || e - Q(w^{1/2}) s\|
  double trace_ws = T::TraceInnerProduct(o->W, minus_s);
  info->normsqrd =
      T::TraceInnerProduct(T::QuadraticRepresentation(o->W, minus_s), minus_s) +
      2 * trace_ws + Rank(*o);

  // TODO(FrankPermenter): replace this heuristic approximation.
  info->norminfd = 1.0 / 3.0 * (trace_ws + Rank(*o));
}

template <>
void GetWeightedSlackEigenvalues(HermitianPsdConstraint<Octonions>* o,
                                 const Ref& y, double c_weight,
                                 WeightedSlackEigenvalues* p) {
  using T = Octonions;
  typename T::Matrix minus_s;
  o->ComputeNegativeSlack(c_weight, y, &minus_s);

  double normsqrd =
      T::TraceInnerProduct(T::QuadraticRepresentation(o->W, minus_s), minus_s);

  // Heuristic approximation based off of inequality:  |x|_1 |x|_{\infty} >=
  // |x|^2_2.
  p->lambda_max = std::fabs(normsqrd) /
                  (1e-15 + std::fabs(T::TraceInnerProduct(o->W, minus_s)));

  // Heuristic.
  p->lambda_min = p->lambda_max * .01;
  p->trace = -T::TraceInnerProduct(o->W, minus_s);
  p->frobenius_norm_squared =
      T::TraceInnerProduct(T::QuadraticRepresentation(o->W, minus_s), minus_s);
}

template <typename T>
void ConstructSchurComplementSystem(HermitianPsdConstraint<T>* o,
                                    bool initialize,
                                    SchurComplementSystem* sys) {
  auto G = &sys->G;
  auto& W = o->W;
  int m = o->constraint_matrices_.size();

  typename T::Matrix AW;
  typename T::Matrix WAW;
  if (initialize) {
    for (int i = 0; i < m; i++) {
      if constexpr (std::is_same<T, Octonions>::value) {
        WAW = T::QuadraticRepresentation(W, o->constraint_matrices_.at(i));
      } else {
        AW = T::Multiply(o->constraint_matrices_.at(i), W);
        WAW = T::Multiply(W, AW);
      }
      for (int j = i; j < m; j++) {
        (*G)(j, i) = o->EvalDualConstraint(j, WAW);
      }
      if constexpr (std::is_same<T, Octonions>::value) {
        sys->AW(i, 0) = o->EvalDualConstraint(i, W);
      } else {
        sys->AW(i, 0) = AW.at(0).trace();
      }
      sys->AQc(i, 0) = o->EvalDualObjective(WAW);
    }
  } else {
    for (int i = 0; i < m; i++) {
      if constexpr (std::is_same<T, Octonions>::value) {
        WAW = T::QuadraticRepresentation(W, o->constraint_matrices_.at(i));
      } else {
        AW = T::Multiply(o->constraint_matrices_.at(i), W);
        WAW = T::Multiply(W, AW);
      }

      for (int j = i; j < m; j++) {
        (*G)(j, i) += o->EvalDualConstraint(j, WAW);
      }

      if constexpr (std::is_same<T, Octonions>::value) {
        sys->AW(i, 0) += o->EvalDualConstraint(i, W);
      } else {
        sys->AW(i, 0) += AW.at(0).trace();
      }
      sys->AQc(i, 0) += o->EvalDualObjective(WAW);
    }
  }
}

template void ConstructSchurComplementSystem(HermitianPsdConstraint<Real>* o,
                                             bool initialize,
                                             SchurComplementSystem* sys);

template void ConstructSchurComplementSystem(HermitianPsdConstraint<Complex>* o,
                                             bool initialize,
                                             SchurComplementSystem* sys);

template void ConstructSchurComplementSystem(
    HermitianPsdConstraint<Quaternions>* o, bool initialize,
    SchurComplementSystem* sys);

template void ConstructSchurComplementSystem(
    HermitianPsdConstraint<Octonions>* o, bool initialize,
    SchurComplementSystem* sys);

template <typename H>
bool UpdateLinearOperator(HermitianPsdConstraint<H>* o, double val, int var,
                          int r, int c, int dim) {
  CONEX_DEMAND(dim < H::HyperComplexDimension(),
               "Complex dimension out of bounds.");
  CONEX_DEMAND(r < o->rank_ && c < o->rank_, "Matrix dimension out of bounds.");
  CONEX_DEMAND(!(val != 0 && r == c && dim > 0),
               "Imaginary components must be skew-symmetric.");

  using T = HermitianPsdConstraint<H>;
  if constexpr (std::is_same<T, Octonions>::value) {
    if (dim >= 3) {
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

template bool UpdateLinearOperator(HermitianPsdConstraint<Complex>* o,
                                   double val, int var, int r, int c, int dim);
template bool UpdateLinearOperator(HermitianPsdConstraint<Real>* o, double val,
                                   int var, int r, int c, int dim);
template bool UpdateLinearOperator(HermitianPsdConstraint<Quaternions>* o,
                                   double val, int var, int r, int c, int dim);
template bool UpdateLinearOperator(HermitianPsdConstraint<Octonions>* o,
                                   double val, int var, int r, int c, int dim);

template <typename H>
bool UpdateAffineTerm(HermitianPsdConstraint<H>* o, double val, int r, int c,
                      int dim) {
  CONEX_DEMAND(dim < H::HyperComplexDimension(),
               "Complex dimension out of bounds.");
  CONEX_DEMAND(r < o->rank_ && c < o->rank_, "Matrix dimension out of bounds.");
  CONEX_DEMAND(!(val != 0 && r == c && dim > 0),
               "Imaginary components must be skew-symmetric.");

  using T = HermitianPsdConstraint<H>;
  if constexpr (std::is_same<T, Octonions>::value) {
    if (dim >= 3) {
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

template bool UpdateAffineTerm(HermitianPsdConstraint<Complex>* o, double val,
                               int r, int c, int dim);
template bool UpdateAffineTerm(HermitianPsdConstraint<Real>* o, double val,
                               int r, int c, int dim);
template bool UpdateAffineTerm(HermitianPsdConstraint<Quaternions>* o,
                               double val, int r, int c, int dim);
template bool UpdateAffineTerm(HermitianPsdConstraint<Octonions>* o, double val,
                               int r, int c, int dim);

}  // namespace conex
