#pragma once
#include <map>
#include <vector>
#include <Eigen/Dense>

namespace conex {



using std::map;
using std::vector;
using Tuple = std::pair<int, int>;
using SparseMatrixTuple = std::vector<std::pair<int, int>>;
using DenseMatrix = Eigen::MatrixXd;

DenseMatrix Symmetrize(const DenseMatrix& X);
Eigen::VectorXd vec(const DenseMatrix& A);
DenseMatrix RandomPSD(int n);
DenseMatrix RandomSym(int n);
DenseMatrix RandomLinear(int n);
vector<SparseMatrixTuple> Partition(DenseMatrix& m);
vector<SparseMatrixTuple> GetRandomTuples(int n, int m);
vector<DenseMatrix> GetRandomDenseMatrices(int order, int num_matrices);
DenseMatrix QuadRep(const DenseMatrix& X, const DenseMatrix& Y);
double NormFro(const DenseMatrix& x);
double min(double x, double y);
inline double max(double x, double y) { return -min(-x, -y); }

struct EigenvalueDecomposition {
  Eigen::MatrixXd eigenvalues;
  Eigen::MatrixXd eigenvectors;
};
EigenvalueDecomposition eig(const Eigen::MatrixXd& x);

} // namespace conex

