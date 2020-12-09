#pragma once
#include <vector>
#include <Eigen/Dense>

namespace conex {
using Matrix = Eigen::Matrix<double, -1, -1>;
bool IsChordal(const Matrix& G, std::vector<int>* chordless_path);
bool IsChordal(const Matrix& G);
bool IsPerfectlyOrdered(const Matrix& G);
Eigen::VectorXd MaximumDegreeVertices(const Matrix& G);
Eigen::PermutationMatrix<-1, -1> EliminationOrdering(const Matrix& G);
}  // namespace conex

