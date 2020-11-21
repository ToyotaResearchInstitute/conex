#pragma once
#include <vector>
#include <Eigen/Dense>

using Clique = std::vector<int>;

struct SparseTriangularMatrix {
  int N;
  std::vector<Clique> path;
  std::vector<int> residual_size;
  // The "fundamental" supernodes. See Figures 4.9 and 3.8 of Chordal Graphs and
  // Semidefinite Optimization.
  std::vector<Eigen::MatrixXd> supernodes;
  std::vector<Eigen::MatrixXd> separator;
  std::vector<int> permutation;
};


std::vector<Clique> Permute(std::vector<Clique>& path, std::vector<int>& permutation);
void Sort(std::vector<Clique>* path);

void IntersectionOfSorted(const std::vector<int>& v1,
                  const std::vector<int>& v2,
                  std::vector<int>* v3);

std::vector<int> UnionOfSorted(const std::vector<int>& x1, const std::vector<int>& x2);


void RunningIntersectionClosure(std::vector<Clique>* path);
SparseTriangularMatrix GetFillInPattern(int N, const std::vector<Clique>& path);
SparseTriangularMatrix MakeSparseTriangularMatrix(int N, const std::vector<Clique>& path);

class TriangularMatrixOperations {
 public:
  using Matrix = SparseTriangularMatrix;
  static Eigen::MatrixXd Multiply(SparseTriangularMatrix& mat, const Eigen::MatrixXd& x);
  static void SetConstant(SparseTriangularMatrix* mat, double val);
  static Eigen::MatrixXd ToDense(const SparseTriangularMatrix& mat);
  static void RescaleColumn(Matrix* mat);
  static void CholeskyInPlace(SparseTriangularMatrix* mat);
  static Eigen::VectorXd ApplyInverse(SparseTriangularMatrix* L, const Eigen::VectorXd& b);
  static Eigen::VectorXd ApplyInverseOfTranspose(SparseTriangularMatrix* L, const Eigen::VectorXd& b);
};

