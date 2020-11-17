#pragma once
#include "conex/tree_gram.h"


struct BlockTriangularOperations {
  static void ApplyBlockInverseInPlace(const SparseTriangularMatrix& L, 
                                                  Eigen::VectorXd* b);
  static void ApplyBlockInverseOfTransposeInPlace(const SparseTriangularMatrix& L, 
                                                  Eigen::VectorXd* b);
  static void BlockCholeskyInPlace(SparseTriangularMatrix* mat);
};
