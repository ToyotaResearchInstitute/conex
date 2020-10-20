#include "gtest/gtest.h"
#include <Eigen/Dense>
#include "conex/debug_macros.h"
#include "conex/eigen_decomp.h"
#include "conex/jordan_matrix_algebra.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using conex::jordan_algebra::eig;
using conex::jordan_algebra::Roots;

template<typename T>
void DUMPA(const T& M) {
  for (const auto& e : M) {
    DUMP(e);
  }
}
VectorXd sort(const VectorXd& x) {
  auto y = x;
  std::sort(y.data(), y.data() + x.rows());
  return y;
}

#if 1
template<typename T>
void DoMultTest() {
  using Element = typename T::Element;
  Element x;
  Element y;
  x.setZero();
  x(0) = 1;
  y.setConstant(3);

  Element z;
  T().Multiply(x,y, &z);
  EXPECT_TRUE((y - z).norm() < 1e-8);
}

TEST(JordanMatrixAlgebra, Scalars) {
  DoMultTest<DivisionAlgebra<1>>();
  DoMultTest<DivisionAlgebra<2>>();
  DoMultTest<DivisionAlgebra<4>>();
  DoMultTest<DivisionAlgebra<8>>();
}

template<typename T>
void DoMatrixTest() {
  using Matrix = typename T::Matrix;
  Matrix A = T::Random();
  Matrix B = T::Random();
  auto W =  T().JordanMult(A, B);

  EXPECT_TRUE(T::IsHermitian(B));
  EXPECT_TRUE(T::IsHermitian(A));
  EXPECT_TRUE(T::IsHermitian(W));

  // Test Jordan identity.
  auto Asqr = T().JordanMult(A, A);
  auto BA = T().JordanMult(A, B);
  auto P1 = T().JordanMult(Asqr, BA);

  auto BAsqr = T().JordanMult(B, Asqr);
  auto P2 = T().JordanMult(A, BAsqr);

  EXPECT_TRUE(T().IsEqual(P1, P2));
}

TEST(JordanMatrixAlgebra, Matrices) {
  DoMatrixTest<Octonions>();
  DoMatrixTest<Quaternions>();
  DoMatrixTest<Complex>();
  DoMatrixTest<Real>();
}

template<typename T>
bool DoQuadAssociativeTest() {
  using Matrix = typename T::Matrix;
  Matrix A = T::Random();
  Matrix B = T::Random();
  Matrix Yref = T().QuadRep(B, A);

  Matrix Y1 = T().MatrixMult(T().MatrixMult(B, A), B);
  Matrix Y2 = T().MatrixMult(B, T().MatrixMult(A, B));
  Y1 = T().ScalarMult(Y1, .5);
  Y2 = T().ScalarMult(Y2, .5);
  Y1 = T().MatrixAdd(Y1, Y2);

  return T().IsEqual(Y1, Yref);
}

TEST(JordanMatrixAlgebra, QuadRep) {
  EXPECT_TRUE(DoQuadAssociativeTest<Quaternions>());
  EXPECT_TRUE(DoQuadAssociativeTest<Complex>());
  EXPECT_TRUE(DoQuadAssociativeTest<Real>());
}

Eigen::MatrixXd Sparsity(const Eigen::MatrixXd x) {
  Eigen::MatrixXd y(x.rows(), x.cols());
  for (int i = 0; i < x.rows(); i++) {
    for (int j = 0; j < x.cols(); j++) {
      if (std::fabs(x(i, j)) > 1e-9) {
        y(i, j) = 1;
      } else {
        y(i, j) = 0;
      }
    }
  }
  return y;
}



TEST(JordanMatrixAlgebra, EigReal) {
  using T = Real;
  auto Q = T::Random();
  Q = T().JordanMult(Q, Q);

  MatrixXd Xs(3, 3);
  for (int i = 0 ; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Xs(i, j) = Q.at(LinIndex(i, j))(0);
    }
  }

  EXPECT_TRUE( (eigenvalues<T>(Q) - sort(eig(Xs).eigenvalues)).norm() < 1e-8);
}

TEST(JordanMatrixAlgebra, Op) {
  using T = Octonions;
  auto Q = T::Random();
  Q = T().JordanMult(Q, Q);

  double normsqr = T().TraceInnerProduct(Q, Q);
  auto eigvals = Roots(T().MinimalPolynomial(Q));

  EXPECT_TRUE(eigvals.minCoeff() > 0);
  EXPECT_TRUE(std::fabs(eigvals.squaredNorm() - normsqr) < 1e-9);
}



template<typename T>
bool TestExponential() {
  auto wsqrt = T::Random();
  auto s = T::Random();
  auto w = T().JordanMult(wsqrt, wsqrt);
  w = T().Identity();
  s = T().MatrixAdd(w, T::ScalarMult(s, .1)); 

  auto y = Geodesic<T>(w, s);

  return (VectorXd(eigenvalues<T>(s).array().exp()) - eigenvalues<T>(y)  ).norm() < 1e-3;
}

TEST(JordanMatrixAlgebra, Exponential) {
  EXPECT_TRUE(TestExponential<Real>());
  EXPECT_TRUE(TestExponential<Complex>());
  EXPECT_TRUE(TestExponential<Quaternions>());
  EXPECT_TRUE(TestExponential<Octonions>());
}
#endif

// 1 + x + x^2 + x^4
template<typename T>
bool TestInfNorm() {
  auto wsqrt = T::Random();
  auto s = T::Random();
  auto w = T().JordanMult(wsqrt, wsqrt);

  auto z = T().QuadRep(wsqrt, s);
  double z_inf_norm = eigenvalues<T>(z).array().abs().maxCoeff();

  double z_inf = NormInfWeighted<T>(w, s);

  return std::fabs(z_inf * z_inf - z_inf_norm*z_inf_norm) < 1e-6;
}

TEST(JordanMatrixAlgebra, InfNorm) {
  EXPECT_TRUE(TestInfNorm<Real>());
  EXPECT_TRUE(TestInfNorm<Complex>());
  EXPECT_TRUE(TestInfNorm<Quaternions>());
  EXPECT_TRUE(TestInfNorm<Octonions>());
}
#if 0
TEST(REAL, EIG) {
for (int kk = 0; kk < 4; kk++) {
  constexpr int n = 13;
  constexpr int d = 4;
  using vect = Eigen::Map<const Eigen::Matrix<double, n*n, 1>>;
  MatrixXd A = Eigen::MatrixXd::Random(n, n);
  MatrixXd At = A.transpose();
  A = A + At;
  MatrixXd B = Eigen::MatrixXd::Random(n, n);
  MatrixXd Bt = B.transpose();
  B = B*Bt;
  
  MatrixXd X = A*B;

  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n); 
  //X = I + X * .15;
  //X = 

  Eigen::MatrixXd M(n*n, d);
  M.col(0) = vect(I.data());
  auto xpow = X;
  for (int i = 1; i < d; i++) {
    M.col(i) = vect(xpow.data());
    xpow = xpow * X;
  }

  MatrixXd G = M.transpose()* M;
  VectorXd f = M.transpose() * vect(xpow.data());

  VectorXd c = G.colPivHouseholderQr().solve(-f);
  MatrixXd G2(d, d);
  VectorXd f2(d);

  VectorXd trace_pow(2*d);  
  trace_pow(0) = n;
  xpow = X;
  for (int i = 1; i < 2*d; i++) {
    trace_pow(i) = xpow.trace();
    xpow = xpow * X;
  }

  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      G2(i, j) = trace_pow(i+j); 
    }
    f2(i) = trace_pow(d + i);
  }
  VectorXd c2 = G2.colPivHouseholderQr().solve(-f2);
  //DUMP(G2);
  //DUMP(G);
  //DUMP(f);
  //DUMP(f2);
  DUMP(Roots(c).maxCoeff());
  DUMP(Roots(c2).maxCoeff());
  DUMP(eig(X).eigenvalues.maxCoeff());
}
  // trace_{ii} <Q(s) e_i, Q(x) e_i> 
  //
  //      tr I,  tr X,   Tr X^2  = tr X^3
  //      tr X   tr X^2, Tr X^3  = tr X^4
  //      tr X^2 tr X^3, Tr X^4  = tr X^5
  //   
  //
  // <AB, AB AB> =  trace BA AB AB
  //             =        A^2 B A B^2
  //                      B A B^2 A^2
  // <AB, AB AB AB> =  trace BA AB AB AB
  //                =  trace  B AB A B^2 A^2


 // <BA, AB> 
 // <BABA, ABABABAB
 //           I          I AB (AB)^2
 //           BA
 //          (BA)^2
 //
 //      32    <(BA)^2, AB>
 //      23    <BA, ABAB>
 //
 //
 //    <e_i, Q(x) Q(s) e_i >
 //    <Q(x) e_i, Q(s) e_i >
 //   
 //
 //    
}
#endif


