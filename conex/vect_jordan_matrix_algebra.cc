#include "conex/vect_jordan_matrix_algebra.h"
#include "conex/debug_macros.h"
#include "conex/eigen_decomp.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

Eigen::VectorXd Vect(const HyperComplexMatrix& x) {
  int n = x.size();
  int d = x.at(0).rows();
  Eigen::MatrixXd B(d, d*n);
  for (int i = 0; i < n; i++) {
    B.block(0, d*i, d, d) = x.at(i);
  }
  Eigen::Map<VectorXd> map(B.data(), n*d*d);
  return map;
}

template<int n>
Eigen::VectorXd MinimalPolynomial(const HyperComplexMatrix& x) {
  using T = MatrixAlgebra<n>;
  int order = x.at(0).rows();
  int dim = Vect(T::Identity(order)).rows();

  Eigen::MatrixXd M(dim, order);
  HyperComplexMatrix xpow =  T::Identity(order);
  for (int i = 0; i < order; i++) {
    M.col(i) = Vect(xpow);
    xpow = T::JordanMultiply(xpow, x);
  }
  return M.colPivHouseholderQr().solve(-Vect(xpow));
}

Eigen::VectorXd Roots(const Eigen::VectorXd& x) {
  Eigen::MatrixXd c(x.rows(), x.rows());
  c.setZero();
  c.bottomRows(1) = -x.transpose();
  c.topRightCorner(x.rows() - 1, x.rows() - 1) = MatrixXd::Identity(x.rows() - 1, x.rows() - 1);
  return conex::jordan_algebra::eig(c).eigenvalues;
}
}


template<int n>
HyperComplexMatrix MatrixAlgebra<n>::Identity(int d) {
  HyperComplexMatrix Z(n);
  Z.at(0) = Eigen::MatrixXd::Identity(d, d);
  for (int i = 1; i < n; i++) {
    Z.at(i) = Eigen::MatrixXd::Zero(d, d);
  }
  return Z;
}

template<int n>
HyperComplexMatrix MatrixAlgebra<n>::Zero(int r, int c) {
  HyperComplexMatrix Z(n);
  for (int i = 0; i < n; i++) {
    Z.at(i) = Eigen::MatrixXd::Zero(r, c);
  }
  return Z;
}

template<int n>
HyperComplexMatrix MatrixAlgebra<n>::Random(int r, int c) {
  HyperComplexMatrix Z(n);
  for (int i = 0; i < n; i++) {
    Z.at(i) = Eigen::MatrixXd::Random(r, c);
  }
  return Z;
}

template<int n>
HyperComplexMatrix MatrixAlgebra<n>::ConjugateTranspose(const HyperComplexMatrix& x) {
  HyperComplexMatrix Z(n);
  Z.at(0) = x.at(0).transpose();
  for (int i = 1; i < n; i++) {
    Z.at(i) = -x.at(i).transpose();
  }
  return Z;
}

template<int n>
HyperComplexMatrix MatrixAlgebra<n>::Multiply(const HyperComplexMatrix& X,
                                              const HyperComplexMatrix& Y) {

  Eigen::MatrixXd M(8, 8);
  Eigen::MatrixXd I(8, 8);
  M << 1,  1,  1,  1,  1,  1,  1,  1, 
       1, -1, -1,  1, -1,  1,  1, -1,
       1,  1, -1, -1, -1, -1,  1,  1, 
       1, -1,  1, -1, -1,  1, -1,  1, 
       1,  1,  1,  1, -1, -1, -1, -1,
       1, -1,  1, -1,  1, -1,  1, -1,
       1, -1, -1,  1,  1, -1, -1,  1, 
       1,  1, -1, -1,  1,  1, -1, -1;

  I << 0,  1,  2,  3,  4,  5,  6,  7,
       1,  0,  3,  2,  5,  4,  7,  6,
       2,  3,  0,  1,  6,  7,  4,  5,
       3,  2,  1,  0,  7,  6,  5,  4,
       4,  5,  6,  7,  0,  1,  2,  3,
       5,  4,  7,  6,  1,  0,  3,  2,
       6,  7,  4,  5,  2,  3,  0,  1,
       7,  6,  5,  4,  3,  2,  1,  0;

  // (A1 A2 A3)(B1 B2 B3)
  HyperComplexMatrix Z = Zero(X.at(0).rows(), Y.at(0).cols());
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (M(i, j) >= 1) {
        Z.at(I(i, j)) += X.at(i) * Y.at(j);
      } else {
        Z.at(I(i, j)) -= X.at(i) * Y.at(j);
      }
    }
  }

  return Z;
}

template<int n>
  typename MatrixAlgebra<n>::Matrix MatrixAlgebra<n>::Add(const Matrix&x, const Matrix& y) {
    assert(x.size() == n);
    assert(y.size() == n);
    Matrix z(n);
    for (int i = 0; i < n; i++) {
      z.at(i) = x.at(i) + y.at(i);
    }
    return z;
  }

template<int n>
  typename MatrixAlgebra<n>::Matrix MatrixAlgebra<n>::ScalarMultiply(const Matrix&x, double s) {
    assert(x.size() == n);
    Matrix z(n);
    for (int i = 0; i < n; i++) {
      z.at(i) = x.at(i).array() * s;
    }
    return z;
  }

template<int n>
  typename MatrixAlgebra<n>::Matrix MatrixAlgebra<n>::JordanMultiply(const Matrix& x, const Matrix& y) {
    assert(x.size() == n);
    assert(y.size() == n);
    return ScalarMultiply(Add(Multiply(x, y), Multiply(y, x)), .5);
  }

template<int n>
  typename MatrixAlgebra<n>::Matrix MatrixAlgebra<n>::QuadraticRepresentation(
      const Matrix& x, const Matrix& y) {
    auto X1 =  ScalarMultiply(JordanMultiply(x, JordanMultiply(x, y)), 2);
    auto X2 =  ScalarMultiply(JordanMultiply(JordanMultiply(x, x), y), -1);
    return Add(X1, X2);
  }

template<int n>
  bool MatrixAlgebra<n>::IsEqual(const Matrix&x, const Matrix& y) {
    for (int i = 0; i < n; i++) {
      if ((x.at(i) - y.at(i)).norm() > 1e-8) {
        return false;
      }
    }
    return true;
  }

template<int n>
  bool MatrixAlgebra<n>::IsHermitian(const Matrix& w) {
    double eps = 1e-12;
    if ((w.at(0) - w.at(0).transpose()).norm() > eps) {
      return false;
    }
    for (int i = 1; i < n; i++) {
      if ((w.at(i) + w.at(i).transpose()).norm() > eps) {
        return false;
      }
    }
    return true;
  }

template<int n>
double MatrixAlgebra<n>::TraceInnerProduct(const Matrix& x, const Matrix& y) {
  double ip = 0;
  for (int i = 0; i < n; i++) {
    ip += x.at(i).cwiseProduct(y.at(i)).colwise().sum().sum();
  }
  return ip;
}


template<int n>
Eigen::VectorXd MatrixAlgebra<n>::Eigenvalues(const Matrix& Q) {
  assert(Q.size() == n);
  return Roots(MinimalPolynomial<n>(Q));
}

template<int n>
typename MatrixAlgebra<n>::Matrix MatrixAlgebra<n>::Orthogonalize(const Matrix& Qn) {
  using T = MatrixAlgebra<n>;
  if (n >= 8) {
    bool is_real_complex_or_quaternion = false;
    assert(is_real_complex_or_quaternion);
  }
  auto Q = Qn;
  int d = Q.at(0).cols();
  for (int i = 0; i < d; i++) {
    double norm = std::sqrt(T::TraceInnerProduct(Q.col(i), Q.col(i)));
    Q.col(i) = T::ScalarMultiply(Q.col(i), 1.0/norm);
    for (int j = i + 1; j < d; j++) {
      auto ip = T::Multiply(T::ConjugateTranspose(Q.col(i)), Q.col(j));
      ip = T::ScalarMultiply(ip, -1);
      Q.col(j)  =  T::Add(Q.col(j),  T::Multiply(Q.col(i), ip));
    }
  }
  return Q;
}









template class MatrixAlgebra<1>;
template class MatrixAlgebra<2>;
template class MatrixAlgebra<4>;
template class MatrixAlgebra<8>;
