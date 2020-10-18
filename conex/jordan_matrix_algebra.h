#pragma once
#include <Eigen/Dense>

constexpr int d = 3;
inline int LinIndex(int i, int j) {
   return j * d + i;
}

template<int n = 8>
class DivisionAlgebra {
 public:
  static constexpr int dim = (.5*(d*d-d))*n + d;
  using Element = Eigen::Matrix<double, n, 1>;
  using Matrix = std::array<Element, d*d>;

  DivisionAlgebra(); 

  void ScalarMult(const Element& x, const Element& y, Element* z);

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

  Eigen::Matrix<int, 8, 8> M;
  Eigen::Matrix<int, 8, 8> I;
};

using Octonions = DivisionAlgebra<8>;
using Quaternions = DivisionAlgebra<4>;
using Complex = DivisionAlgebra<2>;
using Real = DivisionAlgebra<1>;

