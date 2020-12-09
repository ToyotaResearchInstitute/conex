#include "conex/approximate_eigenvalues.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

//    1) Find orthogonal polynomials phi_n with respect to a discrete measure,
//    supported by eigenvalues of A.
//            1) Initialize basis to 1: p_1 = 1
//            2) For i = 1:N
//                  Select p_{N+1} by running Gram-Schmidt on:
//                   {p_1, p_2, \ldots, t p_N}
//               (From three-term recurrence, p_{N+1} is the (normalized)
//               projection of t p_N onto the orthogonal complement of subspace
//               spanned by p_{N-1}, p_{N}.
//
//    2) Find eigenvalues of Jacobi matrix J.
//
//            The three-term recurrence can be written
//              (J - t I) Phi_{n-1}(t) = - phi_n(t) e_n
//
//            Which shows that the zeros of phi_n are eigenvalues of J.
//
//    Example: for the inner-product  <p, q> = <p(A) r_0, p(A) r_0> and
//    symmetric A, reduces to Lanczos method where we use the simplification
//
//                        <q_i(A) r_0, A r_0> =
//
//  Interpretations of truncation:
//    Let f(q) := | q - t^n |. A polynomial q_{n-1} is the best
//    approximation of t^n, i.e., q_{n-1} = argmin f(q) if and only if
//    t^n - q_{n-1} is orthogonal to span{ p_1, \ldots p_{n-1}}.

class JacobiSolver {
 public:
  JacobiSolver(const MatrixXd& A, const MatrixXd& W, const VectorXd& r0, int n)
      : W_(W), r0_(r0), n_(n), powers_of_A_(n + 1) {
    powers_of_A_.at(0) = MatrixXd::Identity(A.rows(), A.rows());
    for (int i = 1; i <= n; i++) {
      powers_of_A_.at(i) = A * powers_of_A_.at(i - 1);
    }
  }

  // Polynomials p and q are in the monomial basis.
  double PolynomialInnerProduct(const Eigen::MatrixXd& p_v,
                                const Eigen::MatrixXd& q_v) {
    // return (EvalPoly(p_v) * EvalPoly(q_v)).trace();

    //   A = WS is diagonalizable since it is similar to the symmetric matrix
    //   W^{1/2} S W^{1/2}. Hence, A = M D inv(M), plying that
    //
    //   Hence, p(A) = M p(D) inv(M)
    //
    //   It follows that
    //   <p(A)^T r0, q(A) r0> = trace r0^T P(A) q(A) r0 = \trace M p q(D) Minv
    //   r0 r0^T >=0
    return (EvalPoly(p_v).transpose() * r0_).dot(EvalPoly(q_v) * W_ * r0_);
  }

  double VectorInnerProduct(const Eigen::MatrixXd& p,
                            const Eigen::MatrixXd& q) {
    // Identify p with p_v and q with q_v.
    return PolynomialInnerProduct(p, q);
  }

  MatrixXd EvalPoly(const Eigen::MatrixXd& p) {
    int n = p.rows();
    MatrixXd y = p(0) * powers_of_A_.at(0);
    for (int i = 1; i < n; i++) {
      y += p(i) * powers_of_A_.at(i);
    }
    return y;
  }

  MatrixXd Eigenvalues() {
    int n = n_;
    VectorXd alpha_v(n);
    VectorXd beta_v(n);
    std::vector<VectorXd> v(n + 1);
    VectorXd v0(n);
    v0.setZero();
    VectorXd one(n);
    one.setZero();
    one(0) = 1;
    double beta = std::sqrt(PolynomialInnerProduct(one, one));

    v.at(0).resize(n);
    v.at(0).setZero();
    v.at(1) = one / beta;

    VectorXd vhat(n);
    for (int j = 1; j < n; j++) {
      VectorXd Avj(n);
      v.at(j).resize(n);
      Avj(0) = 0;
      Avj.tail(n - 1) = v.at(j).head(n - 1);
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
    Avj(0) = 0;
    Avj.tail(n) = v.at(n);
    double alpha = VectorInnerProduct(Avj, v.at(n));
    alpha_v(n - 1) = alpha;

    Eigen::SelfAdjointEigenSolver<MatrixXd> x;
    x.computeFromTridiagonal(alpha_v, beta_v,
                             Eigen::DecompositionOptions::EigenvaluesOnly);
    return x.eigenvalues();
  }

  std::vector<MatrixXd> powers_of_A_;
  VectorXd r0_;
  MatrixXd W_;
  int n_;
};

}  // namespace

Eigen::VectorXd EigenvaluesOfJacobiMatrix(const Eigen::MatrixXd& A,
                                          const Eigen::MatrixXd& W,
                                          const Eigen::VectorXd& r0, int n) {
  JacobiSolver jacobi(A, W, r0, n);
  return jacobi.Eigenvalues();
}

// Performs Lanczo iterations, i.e., given symmetric A and vector r, constructs
// the Jacobi matrix with respect to the inner-product:
//
//   <p, q> =  \langle p(A) r, q(A) r \rangle.
//          = r' Q p(D) q(D) Q^T r
//
//  We identify a polynomial p with the vector v := p(A) r. This way, the
//  the inner-product reduces to the usual dot product. We can also
//  recursively update v since t * p(t) is identified with (t*p)(A) r = A * p(A)
//  r.
Eigen::VectorXd ApproximateEigenvalues(const MatrixXd& A, const MatrixXd& r0,
                                       int num_iter) {
  int n = A.rows();
  VectorXd alpha(num_iter);
  VectorXd beta(num_iter - 1);
  MatrixXd V(n, num_iter);

  V.col(0) = r0 / r0.norm();
  alpha(0) = V.col(0).transpose() * A * V.col(0);
  VectorXd wprev = A * V.col(0) - alpha(0) * V.col(0);

  VectorXd wj;
  for (int j = 1; j < num_iter; j++) {
    auto&& v = V.col(j);
    beta(j - 1) = wprev.norm();
    v = wprev / beta(j - 1);
    alpha(j) = v.transpose() * A * v;
    wprev = A * v - alpha(j) * v - beta(j - 1) * V.col(j - 1);
  }

  Eigen::SelfAdjointEigenSolver<MatrixXd> x;
  x.computeFromTridiagonal(alpha, beta,
                           Eigen::DecompositionOptions::EigenvaluesOnly);
  return x.eigenvalues();
}

double inner_product(const MatrixXd& V, const MatrixXd& U) {
  // return V.col(0).dot(V.col(1));
  return V.col(0).dot(U.col(1));
}

Eigen::VectorXd AsymmetricLanczos(const MatrixXd& WS, const MatrixXd& W,
                                  const MatrixXd& r, int num_iter) {
  int n = WS.rows();
  // Given WS, W, and r,  computes a sequence V_i of num_iter orthogonal
  // polynomials with respect to the inner-product <p, q> :=  r^T W p(WS)  q(WS)
  // r>
  //         =  r^T W M p(D) q(D) inv(M) r
  //         =  r^T W^{1/2} Q p(D) q(D) Q^T W{1/2}
  //
  //  where M =  W^{-1/2} Q  for diagonal D and orthogonal Q since
  //
  //       WS = W^{-1/2} Q D Q^T W^{1/2}
  //
  //  For each iteration, we identify the polynomial p with the
  //  matrix V = [ q(WS)r,  p(WS)^T W r].
  Eigen::MatrixXd V(n, 2);

  //  Diagonal alpha and off diagonal beta of the
  //  tridiagonal symmetric Jacobi matrix:
  VectorXd alpha(num_iter);
  VectorXd beta(num_iter - 1);

  // Temporary variables for the three-term recurrence:
  Eigen::MatrixXd U(n, 2);
  Eigen::MatrixXd Vprev(n, 2);

  // Execute the three term recurrence (equivalent to Gram-Schmidt):
  V.col(1) = r;
  V.col(0) = W * r;
  V = V / std::sqrt(inner_product(V, V));
  Vprev = V;

  U.col(0) = WS * V.col(0);
  U.col(1) = WS.transpose() * V.col(1);

  alpha(0) = inner_product(V, U);
  U = U - alpha(0) * V;

  int cnt = 0;
  for (int j = 1; j < num_iter; j++) {
    beta(j - 1) = inner_product(U, U);
    if (beta(j - 1) < 1e-6) {
      break;
    } else {
      beta(j - 1) = std::sqrt(beta(j - 1));
    }

    Vprev = V;
    V = U / beta(j - 1);
    U.col(0) = WS * V.col(0);
    U.col(1) = WS.transpose() * V.col(1);
    alpha(j) = inner_product(V, U);
    U = U - alpha(j) * V - beta(j - 1) * Vprev;
    cnt++;
  }

  // Return eigenvalues of Jacobi matrix.
  Eigen::SelfAdjointEigenSolver<MatrixXd> x;
  x.computeFromTridiagonal(alpha.head(cnt + 1), beta.head(cnt),
                           Eigen::DecompositionOptions::EigenvaluesOnly);
  return x.eigenvalues();
}

Eigen::VectorXd ApproximateEigenvalues(const Eigen::MatrixXd& WS,
                                       const Eigen::MatrixXd& W,
                                       const Eigen::MatrixXd& r,
                                       int num_iterations, bool compressed) {
  if (compressed) {
    return AsymmetricLanczos(WS, W, r, num_iterations);
  } else {
    return EigenvaluesOfJacobiMatrix(WS, W, r, num_iterations);
  }
}
