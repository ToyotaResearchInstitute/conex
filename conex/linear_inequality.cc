#include "conex/linear_inequality.h"
#include "conex/debug_macros.h"

namespace conex {

using T = LinearInequality;
using Eigen::MatrixXd;
using std::vector;

namespace {

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

}  // namespace

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

void SetDiagonalBlock(T& o, vector<int> r, Eigen::Map<MatrixXd>* data) {
  int i = 0;
  for (auto ri : r) {
    int j = 0;
    for (auto ci : r) {
      (*data)(i, j) = o.GetCoeff(ri, ci);
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

void T::SetOffDiagonal(Eigen::Map<MatrixXd>* data) {}

void T::SetSupernodeDiagonal(Eigen::Map<MatrixXd>* data) { data->setZero(); }

int T::GetCoeff(int i, int j) {
  int row = Find(variables_, i);
  int col = Find(variables_, j);
  return A_.col(row).dot(A_.col(col));
}

void T::BindDiagonalBlock(const DiagonalBlock* data) { diag.push_back(*data); }

void T::BindOffDiagonalBlock(const OffDiagonalBlock* data) {
  if (data->stride != -1) {
    off_diag.push_back(*data);
  } else {
    scatter_block.push_back(*data);
  }
}

template <typename T>
vector<T> InitVector(T* data, int N) {
  vector<T> y;
  for (int i = 0; i < N; i++) {
    y.push_back(data[i]);
  }
  return y;
}

void T::Scatter(const std::vector<int>& r, const std::vector<int>& c,
                double** data) {
  int cnt = 0;
  for (size_t j = 0; j < c.size(); j++) {
    for (size_t i = j; i < r.size(); i++) {
      *data[cnt++] += GetCoeff(r.at(i), c.at(j));
    }
  }
}

void T::UpdateBlocks() {
  for (const auto& d : diag) {
    Eigen::Map<MatrixXd> data(d.data, d.num_vars, d.num_vars);
    if (d.assign) {
      Set(InitVector(d.var_data, d.num_vars),
          InitVector(d.var_data, d.num_vars), &data);
    } else {
      Increment(InitVector(d.var_data, d.num_vars),
                InitVector(d.var_data, d.num_vars), &data);
    }
  }
  for (const auto& d : off_diag) {
    Eigen::Map<MatrixXd> data(d.data, d.num_rows, d.num_cols);
    if (d.assign) {
      SetDiagonalBlock(*this, InitVector(d.row_data, d.num_rows), &data);
    } else {
      Increment(InitVector(d.row_data, d.num_rows),
                InitVector(d.col_data, d.num_cols), &data);
    }
  }

  for (const auto& d : scatter_block) {
    Scatter(InitVector(d.row_data, d.num_rows),
            InitVector(d.col_data, d.num_cols), d.data_pointers);
  }
}

} // namespace conex
