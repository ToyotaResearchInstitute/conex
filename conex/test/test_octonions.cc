#include "gtest/gtest.h"

#include <Eigen/Dense>
#include "conex/debug_macros.h"
using Eigen::VectorXd;

constexpr int n = 8;
constexpr int d = 3;
using Octonion = Eigen::Matrix<double, n, 1>;
using Table = Eigen::Matrix<int, n, n>;
using OctonionMatrix = std::array<Octonion, d*d>;


int LinIndex(int i, int j) {
   return j * d + i;
}

class Octonions {
 public:
  Octonions() {
  M << 1,  1,   1,    1,  1,  1,  1,  1, 
      1,  -1,  -1,   1,  -1, 1,  1,  -1,
      1,  1,   -1,   -1, -1, -1, 1,  1, 
      1,  -1,  1,    -1, -1, 1,  -1, 1, 
      1,  1,   1,    1,  -1, -1, -1, -1,
      1,  -1,  1,    -1, 1,  -1, 1,  -1,
      1,  -1,  -1,   1,  1,  -1, -1, 1, 
      1,  1,   -1,   -1, 1,  1,  -1, -1;

  I << 0, 1, 2,   3,   4,   5,   6,  7,
       1,  0, 3,  2,    5,  4,   7,   6,
       2, 3,  0,   1,    6,   7, 4,  5,
       3,  2,1,    0,    7,  6,   5, 4,
       4,  5, 6,   7,    0,   1,   2,  3,
       5,  4,7,    6,   1,    0, 3,   2,
       6,  7, 4,  5,   2,    3,   0, 1,
       7,  6,  5,   4,   3,   2,    1,  0;
  }


  void mult(Eigen::VectorXd x, Eigen::VectorXd y, Octonion* z) {
    z->setZero();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        int index = I(i, j);
        double gain = M(i, j);
        (*z)(index) += gain * x(i) * y(j) ;
      }
    }
  }

  OctonionMatrix MatrixMult(const OctonionMatrix&x, const OctonionMatrix& y) {
    OctonionMatrix z;
    Octonion temp;
    for (int i = 0; i < d; i++) {
      for (int j = 0; j < d; j++) {
        z.at(LinIndex(i, j)).setZero();
        for (int k = 0; k < d; k++) {
          mult(x.at( LinIndex(i, k)  ), y.at(  LinIndex(k, j) ), &temp);
          z.at(LinIndex(i, j)) += temp;
        }
      }
    }
    return z;
  }

  OctonionMatrix MatrixAdd(const OctonionMatrix&x, const OctonionMatrix& y) {
    OctonionMatrix z;
    Octonion temp;
    for (int i = 0; i < d; i++) {
      for (int j = 0; j < d; j++) {
        z.at(LinIndex(i, j)) = x.at(LinIndex(i, j)) + y.at(LinIndex(i, j));
      }
    }
    return z;
  }
  OctonionMatrix ScalarMult(const OctonionMatrix&x, double s) {
    OctonionMatrix z;
    Octonion temp;
    for (int i = 0; i < d; i++) {
      for (int j = 0; j < d; j++) {
        z.at(LinIndex(i, j)) = x.at(LinIndex(i, j)).array() * s;
      }
    }
    return z;
  }

  OctonionMatrix JordanMult(const OctonionMatrix& x, const OctonionMatrix& y) {
    return ScalarMult(MatrixAdd(MatrixMult(x, y),
                                MatrixMult(y, x)), 
                      .5);
  }

  bool IsEqual(const OctonionMatrix&x, const OctonionMatrix& y) {
    OctonionMatrix z;
    Octonion temp;
    for (int i = 0; i < d; i++) {
      for (int j = 0; j < d; j++) {
        if ((x.at(LinIndex(i, j)) - y.at(LinIndex(i, j))).norm() > 1e-8) {
          return false;
        }
      }
    }
    return true;
  }


  Table M;
  Table I;
};

Octonion Conjugate(const Octonion& x) {
  Octonion y = x;
  y.bottomRows(7).array() *= -1;
  return y;
}

OctonionMatrix Random() {
  OctonionMatrix w;
  Octonion e;
  e(0) = 1;
  e.bottomRows(7).setZero();
  for (int i = 0; i < d; i++) {
    w.at(LinIndex(i, i)) = e * Eigen::MatrixXd::Random(1, 1);
    for (int j = i+1; j < d; j++) {
      w.at(LinIndex(i, j)) = Octonion::Random(); 
      w.at(LinIndex(j, i)) = Conjugate(w.at(LinIndex(i, j)));
    }
  }
  return w;
}

bool IsHermitian(const OctonionMatrix& w) {
  double eps = 1e-12;
  for (int i = 0; i < d; i++) {
    if (w.at(LinIndex(i, i)).bottomRows(7).norm() > eps) {
      return false;
    }
    for (int j = i+1; j < d; j++) {
      if ((w.at(LinIndex(i, j)) - Conjugate(w.at(LinIndex(j, i)))).norm() > eps) {
        return false;
      }
    }
  }
  return true;
}

TEST(JordanAlgebra, OctonionMultIdentity) {
  Octonion x;
  Octonion y;
  x.setZero();
  x(0) = 1;
  y.setConstant(3);

  Octonion z;
  Octonions().mult(x,y, &z);
  EXPECT_TRUE((y - z).norm() < 1e-8);
}


TEST(JordanAlgebra, OctonionMatrices) {
  OctonionMatrix A = Random();
  OctonionMatrix B = Random();
  auto W =  Octonions().JordanMult(A, B);

  EXPECT_TRUE(IsHermitian(B));
  EXPECT_TRUE(IsHermitian(A));
  EXPECT_TRUE(IsHermitian(W));

  // Test Jordan identity.
  auto Asqr = Octonions().JordanMult(A, A);
  auto BA = Octonions().JordanMult(A, B);
  auto P1 = Octonions().JordanMult(Asqr, BA);

  auto BAsqr = Octonions().JordanMult(B, Asqr);
  auto P2 = Octonions().JordanMult(A, BAsqr);

  EXPECT_TRUE(Octonions().IsEqual(P1, P2));
}


