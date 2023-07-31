#pragma once
#include <vector>
#include "conex/RLDLT.h"
#include "conex/triangular_matrix_workspace.h"
#include <Eigen/Dense>

namespace conex {

struct BlockTriangularOperations {
  static void ApplyBlockInverseInPlace(const TriangularMatrixWorkspace& L,
                                       Eigen::VectorXd* b);
  static void ApplyBlockInverseOfTransposeInPlace(
      const TriangularMatrixWorkspace& L, Eigen::VectorXd* b);
  static bool BlockCholeskyInPlace(TriangularMatrixWorkspace* mat);
  static bool BlockLDLTInPlace(
      TriangularMatrixWorkspace* mat,
      std::vector<eigen_stuff::RLDLT<Eigen::Ref<Eigen::MatrixXd>>>* factorization);
  static void ApplyBlockInverseOfMTranspose(
      const TriangularMatrixWorkspace& mat,
      const std::vector<eigen_stuff::RLDLT<Eigen::Ref<Eigen::MatrixXd>>>
          factorization,
      Eigen::VectorXd* y);

  static void ApplyBlockInverseOfMD(
      const TriangularMatrixWorkspace& mat,
      const std::vector<eigen_stuff::RLDLT<Eigen::Ref<Eigen::MatrixXd>>>
          factorization,
      Eigen::VectorXd* y);

  static void SolveInPlaceLDLT(
      const TriangularMatrixWorkspace& mat,
      const std::vector<eigen_stuff::RLDLT<Eigen::Ref<Eigen::MatrixXd>>>
          factorization,
      Eigen::VectorXd* y) {
    ApplyBlockInverseOfMD(mat, factorization, y);
    ApplyBlockInverseOfMTranspose(mat, factorization, y);
  }

  static void SolveInPlaceCholesky(const TriangularMatrixWorkspace& mat,
                                   Eigen::VectorXd* y) {
    ApplyBlockInverseInPlace(mat, y);
    ApplyBlockInverseOfTransposeInPlace(mat, y);
  }
};

}  // namespace conex
