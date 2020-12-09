#pragma once
#include "conex/debug_macros.h"
#include "conex/eigen_decomp.h"
#include <Eigen/Dense>

namespace conex {


class HyperComplexMatrix : public std::vector<Eigen::MatrixXd> {
 public:
  template <typename T>
  class HyperComplexMatrixRef : public std::vector<Eigen::Ref<T>> {
   public:
    template <typename... Args>
    HyperComplexMatrixRef(Args&&... args)
        : std::vector<Eigen::Ref<T>>(args...) {}
    HyperComplexMatrixRef& operator=(const HyperComplexMatrixRef& x) {
      for (unsigned int i = 0; i < x.size(); i++) {
        this->at(i) = x.at(i);
      }
      return (*this);
    };
    HyperComplexMatrixRef& operator=(const HyperComplexMatrix& x) {
      for (unsigned int i = 0; i < x.size(); i++) {
        this->at(i) = x.at(i);
      }
      return (*this);
    };
  };

  HyperComplexMatrix(int n) : std::vector<Eigen::MatrixXd>(n) {}
  HyperComplexMatrix() : std::vector<Eigen::MatrixXd>() {}

  template <typename T>
  HyperComplexMatrix(const HyperComplexMatrixRef<T>& x) {
    for (unsigned int i = 0; i < x.size(); i++) {
      this->push_back(x.at(i));
    }
  }

  void rescale(double scale) {
    for (unsigned int i = 0; i < size(); i++) {
      this->at(i).array() *= scale;
    }
  }

  double norm() const { return std::sqrt(squaredNorm()); }

  Eigen::VectorXd vect() const {
    int n = size();
    int d = at(0).rows();
    Eigen::VectorXd y(n * d * d);
    int offset = 0;
    for (int i = 0; i < n; i++) {
      y.segment(offset, d * d) =
          Eigen::Map<const Eigen::VectorXd>(at(i).data(), d * d);
      offset += d * d;
    }
    return y;
  }

  double squaredNorm() const {
    double y = this->at(0).squaredNorm();
    for (unsigned int i = 1; i < size(); i++) {
      y += this->at(i).squaredNorm();
    }
    return y;
  }

  HyperComplexMatrixRef<Eigen::MatrixXd> col(int j) {
    HyperComplexMatrixRef<Eigen::MatrixXd> y;
    for (unsigned int i = 0; i < size(); i++) {
      y.push_back(this->at(i).col(j));
    }
    return y;
  }

  HyperComplexMatrixRef<const Eigen::MatrixXd> col(int j) const {
    HyperComplexMatrixRef<const Eigen::MatrixXd> y;
    for (unsigned int i = 0; i < size(); i++) {
      y.push_back(this->at(i).col(j));
    }
    return y;
  }
};

template <int n = 8>
class MatrixAlgebra {
 public:
  using Matrix = HyperComplexMatrix;
  static int HyperComplexDimension() { return n; };
  static Matrix Multiply(const Matrix& x, const Matrix& y);
  static Matrix Add(const Matrix& x, const Matrix& y);
  static Matrix Random(int r, int c);
  static Matrix Zero(int r, int c);
  static Matrix Identity(int d);
  static Matrix Orthogonalize(const Matrix& x);
  static Matrix ConjugateTranspose(const Matrix& x);
  static double TraceInnerProduct(const Matrix& x, const Matrix& y);
  static Matrix JordanMultiply(const Matrix& x, const Matrix& y);

  static Matrix ScalarMultiply(const Matrix& x, double s);
  static Matrix QuadraticRepresentation(const Matrix& x, const Matrix& y);
  static Eigen::VectorXd Eigenvalues(const Matrix& x);
  static Eigen::VectorXd ApproximateEigenvalues(const HyperComplexMatrix& A,
                                                const HyperComplexMatrix& r0,
                                                int num_iter);
  static bool IsHermitian(const Matrix& x);
  static bool IsEqual(const Matrix& x, const Matrix& y);
  static Eigen::VectorXd EigenvaluesOfJacobiMatrix(const HyperComplexMatrix& WS,
                                                   const HyperComplexMatrix& W,
                                                   int iter);
  static Eigen::VectorXd ApproximateEigenvalues(const HyperComplexMatrix& WS,
                                                const HyperComplexMatrix& W,
                                                const HyperComplexMatrix& r,
                                                int num_iter);

  static Matrix MakeHermitian(const Matrix& x) {
    return ScalarMultiply(Add(x, ConjugateTranspose(x)), .5);
  }
  double NormInfWeighted(const Matrix& w, const Matrix& s);
};

// Finds the largest eigenvalue of Q(w) Q(s) by first finding the coefficients
// of it's minimal polynomial.  This in turn is the largest eigenvalue of
// Q(w^{1/2}) Q(s) Q(w^{1/2}) = Q( Q(w^{1/2}) s ).
template <typename T>
double NormInfWeighted(const typename T::Matrix& w,
                       const typename T::Matrix& s) {
  int rank = w.at(0).rows();
  int iter = rank * rank;
  Eigen::MatrixXd M(rank * rank * w.size(), iter);
  auto y = T::Identity(rank);

  double k = 1;
  for (int i = 0; i < iter; i++) {
    M.col(i) = y.vect();
    y = T::QuadraticRepresentation(w, T::QuadraticRepresentation(s, y));

    if (i == 0) {
      k = T::TraceInnerProduct(y, y);
      if (k > 1e-8) {
        k = std::sqrt(k);
      } else {
        k = std::sqrt(1e-8);
      }
    }

    y = T::ScalarMultiply(y, 1.0 / k);
  }

  Eigen::MatrixXd G = M.transpose() * M;
  Eigen::VectorXd f = M.transpose() * y.vect();

  Eigen::VectorXd c = G.colPivHouseholderQr().solve(-f);
  double z_inf_sqr_cal = conex::jordan_algebra::Roots(c).maxCoeff() * k;
  return std::sqrt(z_inf_sqr_cal);
}

using Octonions = MatrixAlgebra<8>;
using Quaternions = MatrixAlgebra<4>;
using Complex = MatrixAlgebra<2>;
using Real = MatrixAlgebra<1>;

} // namespace conex
