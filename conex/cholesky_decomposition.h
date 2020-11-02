#include <Eigen/Dense>
#include "conex/debug_macros.h"

using Clique = std::vector<int>;
using Eigen::MatrixXd;
using std::vector;

MatrixXd GetMatrix(int N, const std::vector<Clique>& c, std::vector<MatrixXd> m);

inline std::vector<float> Degree(int N, const std::vector<Clique>& x) {
  vector<float> degree(N);
  vector<bool> simplicial(N);
  for (int i = 0; i < N; i++) {
    degree.at(i) = 0;
    simplicial.at(i) = true;
  }

  int k = 0;
  for (const auto& xi : x) {
    for (const auto& xii : xi) {
      if (degree.at(xii) > 0) {
        simplicial.at(xii) = false;
      }
      degree.at(xii) += xi.size() + 1e-1 * k;
    }

    k++;
  }
  for (int i = 0; i < N; i++) {
    if (simplicial.at(i))  {
      degree.at(i) -= 100*0;
    }
  }
  return degree;
}

void SparseCholeskyDecomposition(const MatrixXd& A, 
                  const std::vector<int>& start,
                  const std::vector<int>& num_cols,
                  const std::vector<std::vector<int>>& root_nodes,
                  MatrixXd* R);


void IntersectionOfSorted(const std::vector<int>& v1, 
                  const std::vector<int>& v2,
                  std::vector<int>* v3);

void DifferenceOfSorted(const std::vector<int>& v1, 
                  const std::vector<int>& v2,
                  std::vector<int>* v3);

