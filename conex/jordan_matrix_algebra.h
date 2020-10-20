#pragma once
#include <Eigen/Dense>
#include "conex/eigen_decomp.h"

constexpr int d = 3;
inline int LinIndex(int i, int j) {
   return j * d + i;
}

template<int n = 8>
class DivisionAlgebra {
 public:
  using Element = Eigen::Matrix<double, n, 1>;
  DivisionAlgebra();
  void Multiply(const Element& x, const Element& y, Element* z);
 private:
  Eigen::Matrix<int, 8, 8> M;
  Eigen::Matrix<int, 8, 8> I;
};

template<int n = 8>
class JordanMatrixAlgebra {
  static_assert(n < 8 || (d <= 3 || n <= 8), "Invalid template parameters.");
 public:
  static constexpr int dim = (.5*(d*d-d))*n + d;
  using Element =  typename DivisionAlgebra<n>::Element;
  using Matrix = std::array<Element, d*d>;

  Matrix MatrixMult(const Matrix&x, const Matrix& y);

  Matrix MatrixAdd(const Matrix&x, const Matrix& y);

  static Matrix ScalarMult(const Matrix&x, double s);

  Matrix JordanMult(const Matrix& x, const Matrix& y);

  Matrix QuadRep(const Matrix& x, const Matrix& y);

  bool IsEqual(const Matrix&x, const Matrix& y);

  static Element Conjugate(const Element& x);

  static Matrix Identity();

  static Matrix Random();

  static Matrix Ones();

  static bool IsHermitian(const Matrix& w);

  double TraceInnerProduct(const Matrix& x, const Matrix& y);

  Matrix Basis(int i, int j, int k);

  Matrix Basis(int dim);

  Eigen::MatrixXd LOperator(const Matrix& Q);

  Eigen::VectorXd Vect(const Matrix&X);

  Eigen::VectorXd MinimalPolynomial(const Matrix& x);
 private:
  DivisionAlgebra<n> division_algebra_;
};

template<typename T>
Eigen::VectorXd eigenvalues(const typename T::Matrix& Q) {
  return conex::jordan_algebra::Roots(T().MinimalPolynomial(Q));
}

using Octonions = JordanMatrixAlgebra<8>;
using Quaternions = JordanMatrixAlgebra<4>;
using Complex = JordanMatrixAlgebra<2>;
using Real = JordanMatrixAlgebra<1>;





