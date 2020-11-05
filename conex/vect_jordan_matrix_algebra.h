#pragma once
#include <Eigen/Dense>


using HyperComplexMatrix = std::vector<Eigen::MatrixXd>;

template<int n = 8>
class MatrixAlgebra {
 public:

  using Matrix = HyperComplexMatrix;
  static Matrix Multiply(const Matrix& x, const Matrix& y);
  static Matrix Add(const Matrix& x, const Matrix& y);
  static Matrix Random(int r, int c);
  static Matrix Zero(int r, int c);
  static Matrix Identity(int d);
  static Matrix ConjugateTranspose(const Matrix& x);
  static double TraceInnerProduct(const Matrix& x, const Matrix& y);
  static Matrix JordanMultiply(const Matrix& x, const Matrix& y);

  static Matrix ScalarMultiply(const Matrix& x, double s);
  static Matrix QuadraticRepresentation(const Matrix& x, const Matrix& y);
  static Eigen::VectorXd Eigenvalues(const Matrix& x);
  static bool IsHermitian(const Matrix& x);
  static bool IsEqual(const Matrix& x, const Matrix& y);
};



using Octonions = MatrixAlgebra<8>;
using Quaternions = MatrixAlgebra<4>;
using Complex = MatrixAlgebra<2>;
using Real = MatrixAlgebra<1>;





