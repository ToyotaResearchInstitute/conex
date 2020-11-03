#include "eigen_decomp.h"

namespace conex {
namespace jordan_algebra {

using Eigen::MatrixXd;

EigenvalueDecomposition eig(const Eigen::MatrixXd& x) {
  //conex::jordan_algebra::SpectralDecompSymmetricMatrices<d> spec;
  //spec.Compute(x);
  Eigen::EigenSolver<Eigen::MatrixXd> spec;
  spec.compute(x);
  EigenvalueDecomposition output;
  output.eigenvalues = spec.eigenvalues().array().real();
  // output.eigenvectors = spec.eigenvectors();
  return output;
}

Eigen::MatrixXd Log(const Eigen::MatrixXd& X) {
  auto d = eig(X);
  Eigen::MatrixXd Y(X.rows(), X.rows());
  Y.setZero();
  for (int i = 0; i < d.eigenvectors.cols(); i++) {
    Y += log(d.eigenvalues(i)) *  d.eigenvectors.col(i) *  d.eigenvectors.col(i).transpose();
  }
  return Y;
}

Eigen::MatrixXd ExpMap(const Eigen::MatrixXd& X) {
  auto d = eig(X);
  Eigen::MatrixXd Y(X.rows(), X.rows());
  Y.setZero();
  for (int i = 0; i < d.eigenvectors.cols(); i++) {
    Y += exp(d.eigenvalues(i)) *  d.eigenvectors.col(i) *  d.eigenvectors.col(i).transpose();
  }
  return Y;
}

Eigen::MatrixXd Sqrt(const Eigen::MatrixXd& X) {
  auto d = eig(X);
  Eigen::MatrixXd Y(X.rows(), X.rows());
  Y.setZero();
  for (int i = 0; i < d.eigenvectors.cols(); i++) {
    Y += sqrt(d.eigenvalues(i)) *  d.eigenvectors.col(i) *  d.eigenvectors.col(i).transpose();
  }
  return Y;
}

Eigen::MatrixXd Mean(const Eigen::MatrixXd& S, const Eigen::MatrixXd& Z) {
  auto Zsqrt = Sqrt(Z);
  Eigen::MatrixXd Zsqrtinv = Zsqrt.inverse();
  return Zsqrt * Sqrt(Zsqrtinv*S*Zsqrtinv) *Zsqrt;
}

std::pair<double, double> SpectrumBounds(const Eigen::MatrixXd& X) {
  std::pair<double, double> y;
  y.first = eig(X).eigenvalues.maxCoeff();
  y.second = eig(X).eigenvalues.minCoeff();
  return y;
}

double SpectralRadius(const Eigen::MatrixXd& X) {
  auto r = SpectrumBounds(X);
  double n = std::fabs(r.first);
  double n2 = std::fabs(r.second);
  if (n2 > n) {
    n = n2;
  }
  return n;
}

double NormInfPowerMethod(Ref* X, Ref* temp1) {
  auto temp2 = X;

  temp1->noalias() = (*temp2)*(*temp2);
  double norm_x_sqr = temp1->trace();

  temp2->noalias() = (*temp1)*(*temp1);
  double norm_xn_sqr = temp2->trace(); 
  // x^4 / x^2
  return sqrt(norm_xn_sqr /  norm_x_sqr);
}

Eigen::VectorXd Roots(const Eigen::VectorXd& x) {
  Eigen::MatrixXd c(x.rows(), x.rows());
  c.setZero();
  c.bottomRows(1) = -x.transpose();
  c.topRightCorner(x.rows() - 1, x.rows() - 1) = MatrixXd::Identity(x.rows() - 1, x.rows() - 1);
  return eig(c).eigenvalues;
}

}  // namespace jordan_algebra
}  // namespace conex
