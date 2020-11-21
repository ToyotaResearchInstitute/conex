#include "conex/equality_constraint.h"
#include "conex/debug_macros.h"
using T = EqualityConstraints;
using std::vector;
using Eigen::MatrixXd;

namespace {

std::vector<int> Relabel(const std::vector<int>& x, const std::vector<int>& labels) {
  std::vector<int> y;
  for (auto& xi : x) {
    y.push_back(labels.at(xi));
  }
  return y;
}

int Find(std::vector<int>& x, int v) {
  int i = 0;
  for (auto xi : x) {
    if (v == xi) {
      return i;
    }
    i++;
  }
  return -1;
}

} // namespace

void T::Set(vector<int> r, vector<int> c, Eigen::Map<MatrixXd>* data) {
  int i = 0;
  for (auto ri : r) {
    int j = 0;
    for (auto ci : c) {
      (*data)(i, j) = GetCoeff(ri, ci);
      j++;
    }
    i++;
  }
}

void T::Increment(vector<int> r, vector<int> c, Eigen::Map<MatrixXd>* data) {
  int i = 0;
  for (auto ri : r) {
    int j = 0;
    for (auto ci : c) {
      (*data)(i, j) += GetCoeff(ri, ci);
      j++;
    }
    i++;
  }
}

void T::SetOffDiagonal(Eigen::Map<MatrixXd>* data) {
  if (cached_supernode_off_diagonal) {
    *data = A_supernode_off_diagonal_;
  } else {
    Set(supernodes_, separators_, data);
    A_supernode_off_diagonal_ = *data;
    cached_supernode_off_diagonal = true;
  }
}

void T::SetSupernodeDiagonal(Eigen::Map<MatrixXd>* data) {
  data->setZero();
}

int T::GetCoeff(int i, int j) {
  int row = Find(dual_variables_, i);
  int col = Find(dual_variables_, j);

  // Equality constraints are on off-diagonal
  // part of KKT system.
  bool both_found = row != -1 && col != -1;
  bool both_not_found = row == -1 && col == -1;
  if (both_found || both_not_found)  {
    return 0;
  }

  if (row != -1) {
    col = Find(variables_, j);
    return A_(row, col);
  } else {
    row = Find(variables_, i);
    return A_(col, row);
  }
}

void T::BindDiagonalBlock(const DiagonalBlock* data) {
  diag.push_back(*data);
}

void T::BindOffDiagonalBlock(const OffDiagonalBlock* data) {
  if (data->stride != -1) {
    off_diag.push_back(*data);
  }
}

template<typename T>
vector<T> InitVector(T* data, int N) {
  vector<T> y;
  for (int i = 0; i < N; i++) {
    y.push_back(data[i]);
  }
  return y;
}

void T::UpdateBlocks() {
  for (const auto& d : diag) {
    Eigen::Map<MatrixXd> data(d.data, d.num_vars, d.num_vars);
    if (d.assign) {
      Set(InitVector(d.var_data, d.num_vars), InitVector(d.var_data, d.num_vars), &data);
    } else {
      Increment(InitVector(d.var_data, d.num_vars), InitVector(d.var_data, d.num_vars), &data);
    }
  }
  for (const auto& d : off_diag) {
    Eigen::Map<MatrixXd> data(d.data, d.num_rows, d.num_cols);
    if (d.assign) {
      Set(InitVector(d.row_data, d.num_rows), InitVector(d.col_data, d.num_cols), &data);
    } else {
      Increment(InitVector(d.row_data, d.num_rows), InitVector(d.col_data, d.num_cols), &data);
    }
  }
}
