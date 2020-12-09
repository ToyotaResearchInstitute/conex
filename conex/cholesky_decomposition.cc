#include "conex/cholesky_decomposition.h"

namespace conex {


namespace {

// Given an ordering, we m
// (1, 2, 3)
//     2, 3)
//     *
//     *

// Elimination Tree
//    Var 1 -  (2, 3, 4, 5)
//    Var 2 -  (3, 4, 5)
//
//    (1, 2)
//    (3, 4)
//

class CliqueMatrix {
  using Matrix = Eigen::MatrixXd;

 public:
  explicit CliqueMatrix(Matrix* x, Matrix* r, std::vector<int> clique_start,
                        std::vector<int> num_rows,
                        std::vector<std::vector<int>> root)
      : ptr_(x),
        r_(r),
        n_(x->rows()),
        clique_start_(clique_start),
        num_rows_(num_rows),
        root_(root) {
    clique_index = 0;
    start_of_next_clique = clique_start.at(1);
    size_ = num_rows.at(0);
    assert(n_ == clique_start_.back() + num_rows_.back());
  }

  //  *  *  *
  //  *  i  -   -   -
  //  * colA  block A
  //
  //  * colAz offDiag   BlockAz
  auto colA() { return ptr_->block(i + 1, i, size(i), 1); }
  int num_root_nodes() {
    if (root_.size() == 0) {
      return 0;
    }
    return root_.at(clique_index).size();
  }
  auto colAz(int k) {
    return ptr_->block(root_.at(clique_index).at(k), i, 1, 1);
  }

  auto blockA() { return ptr_->block(i + 1, i + 1, size(i), size(i)); }

  auto offDiagA(int k) {
    return ptr_->block(root_.at(clique_index).at(k), i + 1, 1, size(i));
  }

  auto offDiag(int k, int kk) {
    return ptr_->block(root_.at(clique_index).at(k),
                       root_.at(clique_index).at(kk), 1, 1);
  }

  auto blockAz(int k) {
    return ptr_->block(root_.at(clique_index).at(k),
                       root_.at(clique_index).at(k), 1, 1);
  }

  auto colR() { return r_->block(i + 1, i, size(i), 1); }

  auto colRz(int k) { return r_->block(root_.at(clique_index).at(k), i, 1, 1); }

  void Increment() {
    i++;
    size_--;
    if (i == start_of_next_clique) {
      clique_index++;
      if (i < n_) {
        size_ = num_rows_.at(clique_index);
      } else {
        size_ = 0;
      }

      if (clique_index < static_cast<int>(clique_start_.size()) - 1) {
        start_of_next_clique = clique_start_.at(clique_index + 1);
      } else {
        start_of_next_clique = n_;
      }
      if (clique_index < static_cast<int>(root_.size())) {
        std::vector<int> temp;
        for (auto e : root_.at(clique_index - 1)) {
          if (e > start_of_next_clique) {
            temp.push_back(e);
          }
        }
        for (auto e : root_.at(clique_index)) {
          temp.push_back(e);
        }

        root_.at(clique_index) = temp;
      }
      //     DUMP(root_.at(clique_index));
      if (i > 0) {
        //        assert(0);
      }
    }
  }

 private:
  int size(int i) { return size_ - 1; }

  Matrix* ptr_;
  Matrix* r_;
  int n_;
  int size_ = 0;
  int i = 0;
  int start_of_next_clique;
  int clique_index = 0;
  std::vector<int> clique_start_;
  std::vector<int> num_rows_;
  std::vector<std::vector<int>> root_;
};

}  // namespace

class Node {
  Eigen::MatrixXd M;
  Eigen::MatrixXd R;

  void Eliminate(){};
};

MatrixXd GetMatrix(int N, const std::vector<Clique>& c,
                   std::vector<MatrixXd> m) {
  MatrixXd M(N, N);
  M.setZero();
  for (unsigned int k = 0; k < c.size(); k++) {
    int i = 0;
    for (auto ci : c.at(k)) {
      int j = 0;
      for (auto cj : c.at(k)) {
        M(ci, cj) += m.at(k)(i, j);
        j++;
      }
      i++;
    }
  }
  return M;
}

using Matrix = MatrixXd;
void SparseCholeskyDecomposition(
    const MatrixXd& A, const std::vector<int>& start,
    const std::vector<int>& num_rows,
    const std::vector<std::vector<int>>& root_nodes, MatrixXd* Rptr) {
  auto& R = *Rptr;

  int n = A.rows();
  Matrix temp = A;
  R.setZero();

  CliqueMatrix Adec(&temp, &R, start, num_rows, root_nodes);
  for (int i = 0; i < n; i++) {
    if (temp(i, i) < 0) {
      assert(0);
    }
    R(i, i) = sqrt(temp(i, i));

    if (i < n - 1) {
      auto&& col = Adec.colA();
      Adec.colR() = 1 / R(i, i) * col;
      Adec.blockA() -= 1 / temp(i, i) * col * col.transpose();

      for (int k = 0; k < Adec.num_root_nodes(); k++) {
        auto&& col = Adec.colA();
        Adec.colRz(k) = 1 / R(i, i) * Adec.colAz(k);

        Adec.blockAz(k) -=
            1 / temp(i, i) * Adec.colAz(k) * Adec.colAz(k).transpose();
        Adec.offDiagA(k) -= 1 / temp(i, i) * Adec.colAz(k) * col.transpose();
      }

      // These are scalar operations.
      for (int k = 0; k < Adec.num_root_nodes(); k++) {
        for (int kk = k + 1; kk < Adec.num_root_nodes(); kk++) {
          Adec.offDiag(kk, k) -=
              1.0 / temp(i, i) * Adec.colAz(kk) * Adec.colAz(k).transpose();
        }
      }
    }

    Adec.Increment();
  }
}

void IntersectionOfSorted(const std::vector<int>& v1,
                          const std::vector<int>& v2, std::vector<int>* v3) {
  v3->clear();
  std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(),
                        back_inserter(*v3));
}

void DifferenceOfSorted(const std::vector<int>& v1, const std::vector<int>& v2,
                        std::vector<int>* v3) {
  v3->clear();
  std::set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(),
                      back_inserter(*v3));
}

} // namespace conex
