#pragma once
#include <Eigen/Dense>

namespace conex {

using Ref = Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>;
void ExponentialMapPadeApproximation(
    const Eigen::Ref<const Eigen::MatrixXd>& arg, Ref* result);

}  // namespace conex
