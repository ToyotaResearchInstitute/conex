// Implements: MaximumCardinalitySearch from p. 293 of
//
//   http://www.seas.ucla.edu/~vandenbe/publications/chordalsdp.pdf
//
// Algorithm is originally from  "Rose, Tarjan, Lueker. Simple Linear-Time
// Algorithms to Test Chordality of Graphs, Test Acyclicity of Hypergraphs, and
// Selectively Reduce Acyclic Hypergraphs."
//
//
#include "conex/elimination_ordering.h"

#include <iostream>
#include <vector>

namespace conex {


template <typename T>
inline T Submatrix(const T& X, const std::vector<int>& indices) {
  T S(indices.size(), indices.size());
  for (unsigned int i = 0; i < indices.size(); i++) {
    for (unsigned int j = 0; j < indices.size(); j++) {
      int i_pos = indices.at(i);
      int j_pos = indices.at(j);
      S(i, j) = X(i_pos, j_pos);
    }
  }
  return S;
}

using Order = Eigen::Matrix<int, -1, 1>;
class MaximumCardinalitySearch {
 public:
  explicit MaximumCardinalitySearch(const Matrix& A)
      : A_(A), N_(A_.rows()), W_(N_), not_eliminated(N_) {
    Initialize();
  }

  Eigen::VectorXd FindOrder() {
    int v = MaxDegreeVertex();
    return FindOrder(v);
  }

  Eigen::VectorXd FindOrder(int v) {
    Eigen::VectorXd order(N_);

    order(N_ - 1) = v;
    for (int i = 1; i < N_; i++) {
      not_eliminated(v) = 0;
      UpdateWeights(v);
      v = ArgMaxWeight();
      order(N_ - 1 - i) = v;
    }
    return order;
  }

 private:
  void UpdateWeights(int v) {
    for (int i = 0; i < N_; i++) {
      if (i == v) {
        continue;
      }
      if (is_adjacent(i, v)) {
        W_(i)++;
      }
    }
  }

  int ArgMaxWeight() {
    double max = -1;
    double pos = -1;
    for (int i = 0; i < N_; i++) {
      if (not_eliminated(i) && Weight(i) >= max) {
        max = Weight(i);
        pos = i;
      }
    }
    return pos;
  }

  Eigen::VectorXd Degree() {
    Eigen::VectorXd sums = A_ * Eigen::MatrixXd::Ones(N_, 1);
    for (int i = 0; i < N_; i++) {
      sums(i) -= A_(i, i);
    }
    return sums;
  }

  int MaxDegreeVertex() {
    auto sums = Degree();
    int argmax = 0;
    int max = sums(0);
    for (int i = 1; i < N_; i++) {
      if (sums(i) > max) {
        argmax = i;
        max = sums(i);
      }
    }
    return argmax;
  }

  void Initialize() {
    not_eliminated.setConstant(1);
    W_ = Eigen::MatrixXd::Zero(N_, 1);
  }

  int is_adjacent(int i, int j) { return A_(i, j) != 0; }

  int Weight(int i) { return W_(i); }

  int IncrementWeight(int i) { return W_(i) += 1; }

 private:
  Matrix A_;
  int N_;
  Eigen::VectorXd W_;
  Eigen::VectorXd not_eliminated;
};

// Returns the submatrix induced by the non-zero elements
// on the first row of (A(node:end, node:end);
auto submat(const Matrix& A, int node, std::vector<int>* indices) {
  int N = A.rows();
  Matrix A1 = A.rightCols(N - node);
  Matrix A2 = A1.bottomRows(N - node);
  std::vector<int>& keep = *indices;
  keep.clear();

  for (int i = 0; i < A2.cols(); i++) {
    if (A2(0, i) != 0) {
      keep.push_back(i);
    }
  }

  Matrix A2_cols(A2.rows(), keep.size());
  int i = 0;
  for (const auto index : keep) {
    A2_cols.col(i++) = A2.col(index);
  }

  Matrix Submat(keep.size(), keep.size());
  i = 0;
  for (const auto index : keep) {
    Submat.row(i++) = A2_cols.row(index);
  }

  for (int i = 0; i < Submat.rows(); i++) {
    Submat(i, i) = 1;
  }
  return Submat;
}

bool IsPerfectlyOrdered(const Matrix& Gdata) {
  int n = Gdata.rows();
  Matrix G(n, n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      G(i, j) = static_cast<int>(Gdata(i, j) != 0);
    }
  }
  bool pass = true;

  std::vector<int> indices;
  for (int i = 0; i < n; i++) {
    auto S = conex::submat(G, i, &indices);
    pass = pass && ((S.rows() * S.rows() - (S * S.transpose()).trace()) == 0);
  }
  return pass;
}

auto ExtractChordlessPath(const std::vector<int>& submat_relative_indices,
                          int basenode,
                          const Eigen::MatrixXd& full_adjacency_matrix) {
  int w = basenode;
  int x = -1;
  int y = -1;
  auto A = full_adjacency_matrix;

  auto indices = submat_relative_indices;
  for (unsigned int i = 0; i < indices.size(); i++) {
    indices.at(i) += w;
  }

  // We construct a chordless path following the proof on p. 294 of
  //   http://www.seas.ucla.edu/~vandenbe/publications/chordalsdp.pdf.
  //
  // Let w be such that adj_{+)(w) is not a clique.
  // Pick non-adjacent neighbors (x, y) of w that satisfy
  //
  // w < x < y,
  //
  // such that x is maximal.
  for (unsigned int i = 0; i < indices.size(); i++) {
    if (static_cast<int>(i) == w) continue;
    for (unsigned int j = i + 1; j < indices.size(); j++) {
      if (static_cast<int>(j) == w) continue;
      int i_pos = indices.at(i);
      int j_pos = indices.at(j);
      if ((A(w, j_pos) != 0) && (A(w, i_pos) != 0)) {
        if (A(i_pos, j_pos) == 0) {
          if (i_pos > x) {
            x = i_pos;
            y = j_pos;
          }
        }
      }
    }
  }

  // Pick z adjacent to x not adjacent to w.
  int z = -1;
  for (int i = 0; i < A.rows(); i++) {
    if (A(i, w) == 0 && A(i, x) > 0) {
      z = i;
      break;
    }
  }

  // Pick v to be the smallest node adjacent to z.
  int v = -1;
  for (int i = 0; i < A.rows(); i++) {
    if (i == w) continue;
    if (i == x) continue;
    if (i == y) continue;
    if (A(i, z) > 0) {
      v = i;
      break;
    }
  }

  // Then, (w, x, y, v) is a chordless path.
  std::vector<int> path;
  path.push_back(w);
  path.push_back(x);
  path.push_back(y);
  path.push_back(v);
  return path;
}

bool IsChordal(const Matrix& G, std::vector<int>* chordless_path) {
  int v = 0;
  auto order = conex::MaximumCardinalitySearch(G).FindOrder(v);
  const int n = G.rows();
  Eigen::PermutationMatrix<-1, -1> P2(n);
  for (int i = 0; i < n; i++) {
    P2.indices()[i] = order(i);
  }
  bool pass = true;
  std::vector<int> submat_indices;
  Eigen::MatrixXd Gp = P2.transpose() * G * P2;
  for (int i = 0; i < n; i++) {
    auto S = conex::submat(Gp, i, &submat_indices);
    pass = pass && ((S.rows() * S.rows() - (S * S.transpose()).trace()) == 0);
    if (!pass) {
      if (chordless_path) {
        auto path = ExtractChordlessPath(submat_indices, i, Gp);
        chordless_path->clear();
        for (auto e : path) {
          chordless_path->push_back(e);
        }
      }
      return false;
    }
  }
  return pass;
}

bool IsChordal(const Matrix& G) { return IsChordal(G, NULL); }

Eigen::PermutationMatrix<-1, -1> EliminationOrdering(const Matrix& G) {
  auto order = conex::MaximumCardinalitySearch(G).FindOrder();

  const int n = G.rows();
  Eigen::PermutationMatrix<-1, -1> P2(n);
  for (int i = 0; i < n; i++) {
    P2.indices()[i] = order(i);
  }
  return P2;
}

Eigen::VectorXd MaximumDegreeVertices(const Matrix& G) {
  int N = G.rows();
  Eigen::VectorXd sums = G * Eigen::MatrixXd::Ones(N, 1);
  std::vector<int> argmax;
  int max = sums(0);
  for (int i = 1; i < N; i++) {
    if (sums(i) >= max) {
      if (sums(i) > max) {
        argmax.clear();
      }
      argmax.push_back(i);
      max = sums(i);
    }
  }

  Eigen::VectorXd output(argmax.size());
  for (unsigned int i = 0; i < argmax.size(); i++) {
    output(i) = argmax.at(i);
  }
  return output;
}
}  // namespace conex

