#pragma once
#include <Eigen/Dense>
#include "conex/debug_macros.h"


class HyperComplexMatrix : public std::vector<Eigen::MatrixXd> {
 public:

  template<typename T>
  class HyperComplexMatrixRef : public std::vector<Eigen::Ref<T>> { 
   public:
    template<typename... Args> 
    HyperComplexMatrixRef(Args&&... args): std::vector<Eigen::Ref<T>>(args...) {}
    HyperComplexMatrixRef &operator= (const HyperComplexMatrixRef& x) {
      for (unsigned int i = 0; i < x.size(); i++) {
        this->at(i) = x.at(i); 
      }
      return (*this);
    };
    HyperComplexMatrixRef &operator= (const HyperComplexMatrix& x) {
      for (unsigned int i = 0; i < x.size(); i++) {
        this->at(i) = x.at(i); 
      }
      return (*this);
    };
  };

  HyperComplexMatrix(int n): std::vector<Eigen::MatrixXd>(n) {}

  template<typename T>
  HyperComplexMatrix(const HyperComplexMatrixRef<T>& x) {
    for (unsigned int i = 0; i < x.size(); i++) {
      this->push_back(x.at(i));
    }
  }

  HyperComplexMatrixRef<Eigen::MatrixXd> col(int j) {  
    HyperComplexMatrixRef<Eigen::MatrixXd> y;
    for (unsigned int i = 0; i < size(); i++){
      y.push_back(this->at(i).col(j));
    }
    return y;
  }

  HyperComplexMatrixRef<const Eigen::MatrixXd> col(int j) const {  
    HyperComplexMatrixRef<const Eigen::MatrixXd> y;
    for (unsigned int i = 0; i < size(); i++){
      y.push_back(this->at(i).col(j));
    }
    return y;
  }


};




template<int n = 8>
class MatrixAlgebra {
 public:

  using Matrix = HyperComplexMatrix;
  static Matrix Multiply(const Matrix& x, const Matrix& y);
  static Matrix Add(const Matrix& x, const Matrix& y);
  static Matrix Random(int r, int c);
  static Matrix Zero(int r, int c);
  static Matrix Identity(int d);
  static Matrix Orthogonalize(const Matrix& x);
  static Matrix ConjugateTranspose(const Matrix& x);
  static double TraceInnerProduct(const Matrix& x, const Matrix& y);
  static Matrix JordanMultiply(const Matrix& x, const Matrix& y);

  static Matrix ScalarMultiply(const Matrix& x, double s);
  static Matrix QuadraticRepresentation(const Matrix& x, const Matrix& y);
  static Eigen::VectorXd Eigenvalues(const Matrix& x);
  static bool IsHermitian(const Matrix& x);
  static bool IsEqual(const Matrix& x, const Matrix& y);
};

//template<int n = 8>
//class MatrixAlgebra {
// public:
//  using Matrix = HyperComplexMatrix;
//  static Matrix Multiply(const Matrix& x, const Matrix& y);
//  static Matrix Add(const Matrix& x, const Matrix& y);
//  static Matrix Random(int r, int c);
//  static Matrix Zero(int r, int c);
//  static Matrix Identity(int d);
//  static Matrix Orthogonalize(const Matrix& x);
//  static Matrix ConjugateTranspose(const Matrix& x);
//
//  static Matrix ScalarMultiply(const Matrix& x, double s);
//  static bool IsHermitian(const Matrix& x);
//  static bool IsEqual(const Matrix& x, const Matrix& y);
//};
//
//template <int n>
//class JordanMatrixAlgebra : public MatrixAlgebra<n> {
//  using HermitianMatrix = HyperComplexMatrix;
//  static HermitianMatrix Random(int r, int c);
//  static HermitianMatrix JordanMultiply(const HermitianMatrix& x, const HermitianMatrix& y);
//  static double TraceInnerProduct(const HermitianMatrix& x, const HermitianMatrix& y);
//  static HermitianMatrix QuadraticRepresentation(const HermitianMatrix& x, const HermitianMatrix& y);
//  static Eigen::VectorXd Eigenvalues(const HermitianMatrix& x);
//};
using Octonions = MatrixAlgebra<8>;
using Quaternions = MatrixAlgebra<4>;
using Complex = MatrixAlgebra<2>;
using Real = MatrixAlgebra<1>;

