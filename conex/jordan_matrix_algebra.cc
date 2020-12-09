#include "jordan_matrix_algebra.h"

template <int n>
DivisionAlgebra<n>::DivisionAlgebra() {
  M << 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1,
      1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1,
      -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1;

  I << 0, 1, 2, 3, 4, 5, 6, 7, 1, 0, 3, 2, 5, 4, 7, 6, 2, 3, 0, 1, 6, 7, 4, 5,
      3, 2, 1, 0, 7, 6, 5, 4, 4, 5, 6, 7, 0, 1, 2, 3, 5, 4, 7, 6, 1, 0, 3, 2, 6,
      7, 4, 5, 2, 3, 0, 1, 7, 6, 5, 4, 3, 2, 1, 0;
}

template <int n>
void DivisionAlgebra<n>::Multiply(const Element& x, const Element& y,
                                  Element* z) {
  z->setZero();
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int index = I(i, j);
      double gain = M(i, j);
      (*z)(index) += gain * x(i) * y(j);
    }
  }
}

template <int n>
typename JordanMatrixAlgebra<n>::Matrix JordanMatrixAlgebra<n>::MatrixMult(
    const Matrix& x, const Matrix& y) {
  Matrix z;
  Element temp;
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      z.at(LinIndex(i, j)).setZero();
      for (int k = 0; k < d; k++) {
        division_algebra_.Multiply(x.at(LinIndex(i, k)), y.at(LinIndex(k, j)),
                                   &temp);
        z.at(LinIndex(i, j)) += temp;
      }
    }
  }
  return z;
}

template <int n>
typename JordanMatrixAlgebra<n>::Matrix JordanMatrixAlgebra<n>::MatrixAdd(
    const Matrix& x, const Matrix& y) {
  Matrix z;
  Element temp;
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      z.at(LinIndex(i, j)) = x.at(LinIndex(i, j)) + y.at(LinIndex(i, j));
    }
  }
  return z;
}

template <int n>
typename JordanMatrixAlgebra<n>::Matrix JordanMatrixAlgebra<n>::ScalarMult(
    const Matrix& x, double s) {
  Matrix z;
  Element temp;
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      z.at(LinIndex(i, j)) = x.at(LinIndex(i, j)).array() * s;
    }
  }
  return z;
}

template <int n>
typename JordanMatrixAlgebra<n>::Matrix JordanMatrixAlgebra<n>::JordanMult(
    const Matrix& x, const Matrix& y) {
  return ScalarMult(MatrixAdd(MatrixMult(x, y), MatrixMult(y, x)), .5);
}

template <int n>
typename JordanMatrixAlgebra<n>::Matrix JordanMatrixAlgebra<n>::QuadRep(
    const Matrix& x, const Matrix& y) {
  auto X1 = ScalarMult(JordanMult(x, JordanMult(x, y)), 2);
  auto X2 = ScalarMult(JordanMult(JordanMult(x, x), y), -1);
  return MatrixAdd(X1, X2);
}

template <int n>
bool JordanMatrixAlgebra<n>::IsEqual(const Matrix& x, const Matrix& y) {
  Matrix z;
  Element temp;
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      if ((x.at(LinIndex(i, j)) - y.at(LinIndex(i, j))).norm() > 1e-8) {
        return false;
      }
    }
  }
  return true;
}

template <int n>
typename JordanMatrixAlgebra<n>::Element JordanMatrixAlgebra<n>::Conjugate(
    const Element& x) {
  Element y = x;
  y.bottomRows(n - 1).array() *= -1;
  return y;
}

template <int n>
typename JordanMatrixAlgebra<n>::Matrix JordanMatrixAlgebra<n>::Identity() {
  auto e = Random();
  e = ScalarMult(e, 0);
  for (int i = 0; i < d; i++) {
    e.at(LinIndex(i, i))(0) = 1;
  }
  return e;
}

template <int n>
typename JordanMatrixAlgebra<n>::Matrix JordanMatrixAlgebra<n>::Random() {
  Matrix w;
  Element e;
  e(0) = 1;
  e.bottomRows(n - 1).setZero();
  for (int i = 0; i < d; i++) {
    w.at(LinIndex(i, i)) = e * Eigen::MatrixXd::Random(1, 1);
    for (int j = i + 1; j < d; j++) {
      w.at(LinIndex(i, j)) = Element::Random();
      w.at(LinIndex(j, i)) = Conjugate(w.at(LinIndex(i, j)));
    }
  }
  return w;
}

template <int n>
typename JordanMatrixAlgebra<n>::Matrix JordanMatrixAlgebra<n>::Ones() {
  Matrix w;
  Element e;
  e(0) = 1;
  e.bottomRows(n - 1).setZero();
  for (int i = 0; i < d; i++) {
    w.at(LinIndex(i, i)) = e;
    for (int j = i + 1; j < d; j++) {
      w.at(LinIndex(i, j)).setConstant(1);
      w.at(LinIndex(j, i)) = Conjugate(w.at(LinIndex(i, j)));
    }
  }
  return w;
}

template <int n>
bool JordanMatrixAlgebra<n>::IsHermitian(const Matrix& w) {
  double eps = 1e-12;
  for (int i = 0; i < d; i++) {
    if (w.at(LinIndex(i, i)).bottomRows(n - 1).norm() > eps) {
      return false;
    }
    for (int j = i + 1; j < d; j++) {
      if ((w.at(LinIndex(i, j)) - Conjugate(w.at(LinIndex(j, i)))).norm() >
          eps) {
        return false;
      }
    }
  }
  return true;
}

template <int n>
double JordanMatrixAlgebra<n>::TraceInnerProduct(const Matrix& x,
                                                 const Matrix& y) {
  auto w = JordanMult(x, y);
  double ip = 0;
  for (int i = 0; i < d; i++) {
    ip += w.at(LinIndex(i, i))(0);
  }
  return ip;
}

template <int n>
typename JordanMatrixAlgebra<n>::Matrix JordanMatrixAlgebra<n>::Basis(int i,
                                                                      int j,
                                                                      int k) {
  auto w = ScalarMult(Random(), 0);
  double val = 1;
  if (i != j) {
    val = 1.0 / std::sqrt(2);
  }
  w.at(LinIndex(i, j))(k) = val;

  if (i != j) {
    if (k > 0) {
      val *= -1;
    }
    w.at(LinIndex(j, i))(k) = val;
  }
  return w;
}

template <int n>
typename JordanMatrixAlgebra<n>::Matrix JordanMatrixAlgebra<n>::Basis(int dim) {
  if (dim < d) {
    return Basis(dim, dim, 0);
  }
  int off_pos = (dim - d) / n;
  int off_num = (dim - d) % n;
  if (off_pos == 0) {
    return Basis(0, 1, off_num);
  }
  if (off_pos == 2) {
    return Basis(0, 2, off_num);
  }
  if (off_pos == 1) {
    return Basis(1, 2, off_num);
  }
  assert(0);
}

template <int n>
Eigen::MatrixXd JordanMatrixAlgebra<n>::LOperator(const Matrix& Q) {
  Eigen::MatrixXd D(dim, dim);
  for (int i = 0; i < dim; i++) {
    auto Mi = Basis(i);
    for (int j = i; j < dim; j++) {
      auto Mj = Basis(j);
      D(j, i) = TraceInnerProduct(Mj, JordanMult(Q, Mi));
      D(i, j) = TraceInnerProduct(Mj, JordanMult(Q, Mi));
    }
  }
  return D;
}

template <int n>
Eigen::VectorXd JordanMatrixAlgebra<n>::Vect(const Matrix& X) {
  Eigen::VectorXd y(dim);
  for (int i = 0; i < dim; i++) {
    y(i) = TraceInnerProduct(Basis(i), X);
  }
  return y;
}

template <int n>
Eigen::VectorXd JordanMatrixAlgebra<n>::MinimalPolynomial(const Matrix& x) {
  Matrix xsqr = JordanMult(x, x);
  Matrix xcub = JordanMult(xsqr, x);
  Eigen::MatrixXd M(dim, 3);
  M.col(0) = Vect(Identity());
  M.col(1) = Vect(x);
  M.col(2) = Vect(xsqr);
  return M.colPivHouseholderQr().solve(-Vect(xcub));
}

template class JordanMatrixAlgebra<1>;
template class JordanMatrixAlgebra<2>;
template class JordanMatrixAlgebra<4>;
template class JordanMatrixAlgebra<8>;
template class DivisionAlgebra<1>;
template class DivisionAlgebra<2>;
template class DivisionAlgebra<4>;
template class DivisionAlgebra<8>;
