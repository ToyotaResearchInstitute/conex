#include "conex/test/cholesky_decomposition.h"

namespace {

class CliqueMatrix {
  using Matrix = Eigen::MatrixXd;

 public:
  explicit CliqueMatrix(Matrix* x, Matrix* r, 
                        std::vector<int> clique_start,
                        std::vector<int> num_rows,
                        std::vector<int> num_cols) : ptr_(x), r_(r), n_(x->rows()),
    clique_start_(clique_start),
    num_rows_(num_rows),
    num_cols_(num_cols)  {
    clique_index = 0;
    start_of_next_clique = clique_start.at(1);
    end_of_current_clique = num_cols.at(0);
  }

  auto colA() {
    return ptr_->block(i+1, i, size(i), 1);
  }

  auto blockA() {
    return  ptr_->block(i+1, i+1, size(i), size(i));
  }

  auto colR() {
    return  r_->block(i+1, i,  size(i), 1);
  }

  void Increment() {
    i++;
    if (i == start_of_next_clique) {
      clique_index++;
      if (clique_index < static_cast<int>(clique_start_.size())) {
        end_of_current_clique = clique_start_.at(clique_index) + num_cols_.at(clique_index);
      }

      if (clique_index < static_cast<int>(clique_start_.size())-1) {
        start_of_next_clique = clique_start_.at(clique_index+1);
      } else {
        start_of_next_clique = n_;
      }
    }
  }

 private:
  int size(int i) {
    return end_of_current_clique-1-i;
  }

  Matrix* ptr_;
  Matrix* r_;
  int n_;
  int i = 0;
  int start_of_next_clique;
  int clique_index = 0;
  int end_of_current_clique = 0;
  std::vector<int> clique_start_;
  std::vector<int> num_cols_;
  std::vector<int> num_rows_;
};

}


MatrixXd GetMatrix(int N, const std::vector<Clique>& c, std::vector<MatrixXd> m) {
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
void SparseCholeskyDecomposition(const MatrixXd& A, 
                  const std::vector<int>& start,
                  const std::vector<int>& num_rows,
                  const std::vector<int>& num_cols,
                  MatrixXd* Rptr) {
  auto& R = *Rptr;

  int n = A.rows();
  Matrix temp = A;
  R.setZero();

  CliqueMatrix Adec(&temp, &R, start, num_rows, num_cols);
  for (int i = 0; i < n; i++) {
    R(i, i) = sqrt(temp(i, i));

    if (i < n-1) {
      auto&& col = Adec.colA(); 
      Adec.colR() = 1/R(i, i)*col;
      Adec.blockA() -= 1/temp(i, i) * col * col.transpose();
    }

    Adec.Increment();
  }
}
