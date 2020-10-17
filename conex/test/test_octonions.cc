#include "gtest/gtest.h"

#include <Eigen/Dense>
#include "conex/debug_macros.h"
using Eigen::VectorXd;

constexpr int n = 8;
constexpr int d = 3;
using Table = Eigen::Matrix<int, n, n>;


int LinIndex(int i, int j) {
   return j * d + i;
}

template<int n = 8>
class DivisionAlgebra {
 public:
  using Octonion = Eigen::Matrix<double, n, 1>;
  using OctonionMatrix = std::array<Octonion, d*d>;

  DivisionAlgebra() {
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
          mult(x.at(LinIndex(i, k)), y.at(LinIndex(k, j)), &temp);
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

  static Octonion Conjugate(const Octonion& x) {
    Octonion y = x;
    y.bottomRows(n-1).array() *= -1;
    return y;
  }

  static OctonionMatrix Random() {
    OctonionMatrix w;
    Octonion e;
    e(0) = 1;
    e.bottomRows(n-1).setZero();
    for (int i = 0; i < d; i++) {
      w.at(LinIndex(i, i)) = e * Eigen::MatrixXd::Random(1, 1);
      for (int j = i+1; j < d; j++) {
        w.at(LinIndex(i, j)) = Octonion::Random(); 
        w.at(LinIndex(j, i)) = Conjugate(w.at(LinIndex(i, j)));
      }
    }
    return w;
  }

  static bool IsHermitian(const OctonionMatrix& w) {
    double eps = 1e-12;
    for (int i = 0; i < d; i++) {
      if (w.at(LinIndex(i, i)).bottomRows(n-1).norm() > eps) {
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


  Table M;
  Table I;
};

using Octonions = DivisionAlgebra<8>;
using Quaternions = DivisionAlgebra<4>;
using Complex = DivisionAlgebra<2>;
using Real = DivisionAlgebra<1>;

template<typename T>
void DoMultTest() {
  using Octonion = typename T::Octonion;
  Octonion x;
  Octonion y;
  x.setZero();
  x(0) = 1;
  y.setConstant(3);

  Octonion z;
  T().mult(x,y, &z);
  EXPECT_TRUE((y - z).norm() < 1e-8);
}

TEST(JordanAlgebra, OctonionMultIdentity) {
  DoMultTest<Octonions>();
  DoMultTest<Quaternions>();
  DoMultTest<Complex>();
  DoMultTest<Real>();
}


template<typename T>
void DoMatrixTest() {
  using OctonionMatrix = typename T::OctonionMatrix;
  OctonionMatrix A = T::Random();
  OctonionMatrix B = T::Random();
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


TEST(JordanAlgebra, OctonionMatrices) {
  using OctonionMatrix = Octonions::OctonionMatrix;
  DoMatrixTest<Octonions>();
  DoMatrixTest<Quaternions>();
  DoMatrixTest<Complex>();
  DoMatrixTest<Real>();
}


