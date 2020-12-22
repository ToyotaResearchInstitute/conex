#include "exponential_map_pade.h"
#include <cmath>
#include <complex>
#include "debug_macros.h"

namespace conex {

using Eigen::MatrixXd;
// TODO(FrankPermenter): Remove dynamic memory allocation.
void ComputeWeightedPowers(const MatrixXd& A, MatrixXd* U, MatrixXd* V) {
  const double b[] = {120, 60, 12, 1};
  auto& Asquared = *V;
  Asquared = A * A;

  const MatrixXd tmp =
      b[3] * Asquared + b[1] * MatrixXd::Identity(A.rows(), A.cols());
  U->noalias() = A * tmp;

  (*V).array() *= b[2];
  V->diagonal().array() += b[0];
}

void ExponentialMapPadeApproximation(
    const Eigen::Ref<const Eigen::MatrixXd>& arg, Ref* result) {
  MatrixXd odd_powers(arg.rows(), arg.rows());
  MatrixXd even_powers(arg.rows(), arg.rows());
  ComputeWeightedPowers(arg, &odd_powers, &even_powers);
  MatrixXd numer = odd_powers + even_powers;
  MatrixXd denom = -odd_powers + even_powers;
  // TODO(FrankPermenter): Use inplace decomposition.
  *result = denom.partialPivLu().solve(numer);
}

}  // namespace conex
