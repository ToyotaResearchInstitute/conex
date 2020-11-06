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



template<int d>
Eigen::VectorXd MatrixAlgebra<d>::ApproximateEigenvalues(const HyperComplexMatrix& A, 
                                       const HyperComplexMatrix& r0, int num_iter) {

  using T = MatrixAlgebra<d>;
  assert(T::IsHermitian(A));
  VectorXd alpha(num_iter);
  VectorXd beta(num_iter - 1);
  std::vector<HyperComplexMatrix> V(num_iter); 

  V.at(0) = T::ScalarMultiply(r0, 1.0/r0.norm());

  auto temp = T::Multiply(T::ConjugateTranspose(V.at(0)),  T::Multiply(A, V.at(0)));
  alpha(0) = temp.at(0)(0, 0);
  auto wprev = T::Add(T::Multiply(A, V.at(0)), 
                      T::ScalarMultiply(V.at(0), -alpha(0)));

  VectorXd wj;
  for (int j = 1; j < num_iter; j++) {
    auto && v = V.at(j);
    beta(j -1) = wprev.norm();
    v = T::ScalarMultiply(wprev, 1.0/beta(j - 1));
    // alpha(j) = v.transpose() * A * v;
    alpha(j) = T::Multiply(T::ConjugateTranspose(v),  T::Multiply(A, v)).at(0)(0, 0);
    temp = T::Multiply(T::ConjugateTranspose(v),  T::Multiply(A, v));

    //wprev = A*v - alpha(j) * v - beta(j - 1) * V.col(j-1);
    wprev = T::Add(T::Multiply(A, v), 
                   T::ScalarMultiply(v, -alpha(j)));
    wprev = T::Add(wprev, 
                   T::ScalarMultiply(V.at(j-1), -beta(j - 1)));
  }

  Eigen::SelfAdjointEigenSolver<MatrixXd> x;
      x.computeFromTridiagonal(alpha, beta,
      Eigen::DecompositionOptions::EigenvaluesOnly);
  return x.eigenvalues();
}


template<int d>
class JacobiSolver {
  using T = MatrixAlgebra<d>;
  using Matrix = typename MatrixAlgebra<d>::Matrix;
 public:
  JacobiSolver(const Matrix& A,  
               const Matrix& r0, int n) :  r0_(r0), n_(n), powers_of_A_(n + 1) {
    int order = A.at(0).rows();
    powers_of_A_.at(0) = T::Identity(order);
    for (int i = 1; i <= n; i++) {
      powers_of_A_.at(i) = T::Multiply(A, powers_of_A_.at(i-1));
    }
  }

  // Polynomials p and q are in the monomial basis.
  double PolynomialInnerProduct(const Eigen::MatrixXd& p_v, const Eigen::MatrixXd& q_v) {
    // return T::TraceInnerProduct(EvalPoly(p_v), EvalPoly(q_v));

    //   A = WS is diagonalizable since it is similar to the symmetric matrix W^{1/2} S W^{1/2}.
    //   Hence, A = M D inv(M), plying that
    //
    //   Hence, p(A) = M p(D) inv(M) 
    //
    //   It follows that 
    //   <p(A)^T r0, q(A) r0> = trace r0^T P(A) q(A) r0 = \trace M p q(D) Minv r0 r0^T >=0
    auto temp = T::Multiply(EvalPoly(p_v), EvalPoly(q_v));
    //auto ip = T::Multiply(T::Multiply( T::ConjugateTranspose(r0_),  temp), r0_);
    auto ip = T::Multiply(T::ConjugateTranspose(r0_), T::Multiply(temp, r0_));
    return ip.at(0)(0,0);
  }

  double VectorInnerProduct(const MatrixXd& p, const MatrixXd& q) {
    // Identify p with p_v and q with q_v.
    return PolynomialInnerProduct(p , q);
  }

  Matrix EvalPoly(const Eigen::MatrixXd& p) {
    int n = p.rows();
    auto y = T::ScalarMultiply(powers_of_A_.at(0), p(0));
    for (int i = 1; i < n; i++) {
      y = T::Add(y, T::ScalarMultiply(powers_of_A_.at(i), p(i)));
    }
    return y;
  }

  MatrixXd Eigenvalues() {
    int n = n_;
    VectorXd alpha_v(n);
    VectorXd beta_v(n);
    std::vector<VectorXd> v(n+1);
    VectorXd v0(n); v0.setZero();
    VectorXd one(n); one.setZero(); one(0) = 1;
    double beta = std::sqrt(PolynomialInnerProduct(one, one));

    v.at(0).resize(n); v.at(0).setZero();
    v.at(1) = one / beta;

    VectorXd vhat(n);
    for (int j = 1; j < n; j++) {
      VectorXd Avj(n); v.at(j).resize(n);
      Avj(0) = 0; Avj.tail(n - 1) = v.at(j).head(n - 1);
      double alpha = VectorInnerProduct(Avj, v.at(j));
      vhat = Avj - alpha * v.at(j) - beta * v.at(j - 1);
      beta = std::sqrt(VectorInnerProduct(vhat, vhat));
      if (j < n) {
        v.at(j + 1) = vhat / beta;
      }
      beta_v(j - 1) = beta;
      alpha_v(j - 1) = alpha;
    }

    VectorXd Avj(n + 1);
    Avj(0) = 0; Avj.tail(n) = v.at(n);
    double alpha = VectorInnerProduct(Avj, v.at(n));
    alpha_v(n - 1) = alpha;

    DUMP(alpha_v);
    DUMP(beta_v);
    Eigen::SelfAdjointEigenSolver<MatrixXd> x;
        x.computeFromTridiagonal(alpha_v, beta_v,
        Eigen::DecompositionOptions::EigenvaluesOnly);
    return x.eigenvalues();
  }

  std::vector<Matrix> powers_of_A_;
  Matrix r0_;
  int n_;
};


template<int d>
Eigen::VectorXd MatrixAlgebra<d>::EigenvaluesOfJacobiMatrix(const HyperComplexMatrix& A, 
                                          const HyperComplexMatrix& r0,  int n) {
  JacobiSolver<d> jacobi(A, r0, n);
  return jacobi.Eigenvalues();
}


template class MatrixAlgebra<1>;
template class MatrixAlgebra<2>;
template class MatrixAlgebra<4>;
template class MatrixAlgebra<8>;
