#pragma once
#include <memory>
#include <Eigen/Dense>
#include "conex/newton_step.h"

namespace conex {
namespace jordan_algebra {

struct  EigenvalueDecomposition {
  Eigen::MatrixXd eigenvalues;
  Eigen::MatrixXd eigenvectors;
};

Eigen::MatrixXd Log(const Eigen::MatrixXd& X);
EigenvalueDecomposition eig(const Eigen::MatrixXd& x);
Eigen::MatrixXd Sqrt(const Eigen::MatrixXd& X);
Eigen::MatrixXd Mean(const Eigen::MatrixXd& S, const Eigen::MatrixXd& Z);
Eigen::MatrixXd ExpMap(const Eigen::MatrixXd& X);
double SpectralRadius(const Eigen::MatrixXd& X);
double NormInfPowerMethod(Ref*, Ref*);

std::pair<double, double> SpectrumBounds(const Eigen::MatrixXd& X);

Eigen::VectorXd Roots(const Eigen::VectorXd& x);

}  // namespace jordan_algebra
}  // namespace conex
