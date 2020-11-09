#include "gtest/gtest.h"
#include <Eigen/Dense>
#include "conex/debug_macros.h"
#include "conex/eigen_decomp.h"
#include "conex/jordan_matrix_algebra.h"

namespace conex {

using Eigen::VectorXd;
using Eigen::MatrixXd;
using conex::jordan_algebra::eig;
using conex::jordan_algebra::Roots;

VectorXd sort(const VectorXd& x) {
  auto y = x;
  std::sort(y.data(), y.data() + x.rows());
  return y;
}

template<typename T>
void DoMultTest() {
  using Element = typename T::Element;
  Element id;
  Element y;
  id.setZero();
  id(0) = 1;
  y.setConstant(3);

  Element z;
  T().Multiply(id, y, &z);
  EXPECT_TRUE((y - z).norm() < 1e-8);
}

TEST(JordanMatrixAlgebra, Scalars) {
  DoMultTest<DivisionAlgebra<1>>();
  DoMultTest<DivisionAlgebra<2>>();
  DoMultTest<DivisionAlgebra<4>>();
  DoMultTest<DivisionAlgebra<8>>();
}

template<typename T>
void DoVerifyJordanIdentity() {
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
  DoVerifyJordanIdentity<Octonions>();
  DoVerifyJordanIdentity<Quaternions>();
  DoVerifyJordanIdentity<Complex>();
  DoVerifyJordanIdentity<Real>();
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

TEST(JordanMatrixAlgebra, HermitianRealMatchesEigen) {
  using T = Real;
  auto Q = T::Random();
  Q = T().JordanMult(Q, Q);

  MatrixXd Xs(3, 3);
  for (int i = 0 ; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Xs(i, j) = Q.at(LinIndex(i, j))(0);
    }
  }

  EXPECT_TRUE((eigenvalues<T>(Q) - sort(eig(Xs).eigenvalues)).norm() < 1e-8);
}

template<typename T>
void DoEigenvalueTests() {
  auto Q = T::Random();
  Q = T().JordanMult(Q, Q);

  double normsqr = T().TraceInnerProduct(Q, Q);
  auto eigvals = eigenvalues<T>(Q);

  EXPECT_TRUE(eigvals.minCoeff() > 0);
  EXPECT_TRUE(std::fabs(eigvals.squaredNorm() - normsqr) < 1e-9);

  auto I = T::Identity();
  eigvals = sort(eigenvalues<T>(I));
  for (int i = 0; i < eigvals.rows(); i++) {
    EXPECT_NEAR(eigvals(i), 1, 1e-8);
  }
}

TEST(JordanMatrixAlgebra, EigenvaluePropertiesReal) {
  DoEigenvalueTests<Real>();
}
TEST(JordanMatrixAlgebra, EigenvaluePropertiesComplex) {
  DoEigenvalueTests<Complex>();
}

TEST(JordanMatrixAlgebra, EigenvaluePropertiesQuaternion) {
  DoEigenvalueTests<Quaternions>();
}
TEST(JordanMatrixAlgebra, EigenvaluePropertiesOctonion) {
  DoEigenvalueTests<Octonions>();
}

// Computes Q(w^{1/2}) exp (Q(w^{1/2}) s) from a power series.
template<typename T>
bool DoTestGeodesicPowerSeries() {
  auto wsqrt = T::Random();
  auto s = T::Random();
  auto w = T().JordanMult(wsqrt, wsqrt);
  w = T().Identity();
  s = T().MatrixAdd(w, T::ScalarMult(s, .1));

  auto y = Geodesic<T>(w, s);
  return (VectorXd(eigenvalues<T>(s).array().exp()) - eigenvalues<T>(y)).norm() < 1e-3;
}

TEST(JordanMatrixAlgebra, GeodesicPowerSeries) {
  EXPECT_TRUE(DoTestGeodesicPowerSeries<Real>());
  EXPECT_TRUE(DoTestGeodesicPowerSeries<Complex>());
  EXPECT_TRUE(DoTestGeodesicPowerSeries<Quaternions>());
  EXPECT_TRUE(DoTestGeodesicPowerSeries<Octonions>());
}

// Computes the norm |Q(w^{1/2}) s|_{inf} from the eigenvalues of
// the product Q(w)Q(s) of the quadratic representations Q(w) Q(s).
template<typename T>
bool TestInfNorm() {
  auto wsqrt = T::Random();
  auto s = T::Random();
  auto w = T().JordanMult(wsqrt, wsqrt);

  auto z = T().QuadRep(wsqrt, s);
  double z_inf_norm = eigenvalues<T>(z).array().abs().maxCoeff();

  double z_inf = NormInfWeighted<T>(w, s);

  EXPECT_LE(std::fabs(z_inf * z_inf - z_inf_norm*z_inf_norm), 1e-6);
  return std::fabs(z_inf * z_inf - z_inf_norm*z_inf_norm) < 1e-6;
}

TEST(JordanMatrixAlgebra, InfNormFromQuadRep) {
  EXPECT_TRUE(TestInfNorm<Real>());
  EXPECT_TRUE(TestInfNorm<Complex>());
  EXPECT_TRUE(TestInfNorm<Quaternions>());
  EXPECT_TRUE(TestInfNorm<Octonions>());
}

} // namespace conex

