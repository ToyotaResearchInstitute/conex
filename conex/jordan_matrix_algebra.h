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
  static int Rank() { return d; };

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


inline int factorial(int n) {
  int y = 1;
  for (int i = 2; i <= n; i++) {
    y *= i; 
  }
  return y;
}

template<typename T>
typename T::Matrix Geodesic(const typename T::Matrix& w, 
                            const typename T::Matrix& s) {
  auto y1 = w; 
  auto y2 = T().QuadRep(w, s);
  auto y = T().MatrixAdd(y1, y2);
  for (int i = 1; i < 6; i++) {
    y1 = T().QuadRep(w , T().QuadRep(s , y1));
    y2 = T().QuadRep(w , T().QuadRep(s , y2));
    y =  T().MatrixAdd(y, 
                          T().MatrixAdd( 
                             T::ScalarMult(y1, 1.0/factorial(2*i)),
                             T::ScalarMult(y2, 1.0/factorial(2*i+1)))
                      );
  }
  return y;
}

template<typename T>
double NormInfWeighted(const typename T::Matrix& w, 
                            const typename T::Matrix& s) {

  // |Q(w^{1/2}) s - e|_{\inf}

  // Computes |Q(w^{1/2}) s|_{\inf}

  // int iter = T::Rank() * T::Rank();
  int iter = 2 * T::Rank();
  Eigen::MatrixXd M(T::dim, iter);
  auto y = T::Identity();

  double k = 1;
  for (int i = 0; i < iter; i++) {
    M.col(i) = T().Vect(y);
    y = T().QuadRep(w , T().QuadRep(s , y));

    if (i == 0) {
      k = T().TraceInnerProduct(y, y);
      if (k > 1e-8) {
        k = std::sqrt(k);
      } else {
        k = std::sqrt(1e-8);
      }
    }

    y = T().ScalarMult(y, 1.0/k);
  }

  Eigen::MatrixXd G = M.transpose() * M;
  Eigen::VectorXd f = M.transpose() * T().Vect(y);

  Eigen::VectorXd c = G.colPivHouseholderQr().solve(-f);
  double z_inf_sqr_cal = conex::jordan_algebra::Roots(c).maxCoeff() * k;
  return std::sqrt(z_inf_sqr_cal);
}




using Octonions = JordanMatrixAlgebra<8>;
using Quaternions = JordanMatrixAlgebra<4>;
using Complex = JordanMatrixAlgebra<2>;
using Real = JordanMatrixAlgebra<1>;





