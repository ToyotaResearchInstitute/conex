#pragma once
#include <Eigen/Dense>
using Ref = Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>;
void MatrixExponential(const Eigen::Ref<const Eigen::MatrixXd>& arg,
                       Ref* result);
