#include "conex/kkt_assembler.h"
#include "conex/debug_macros.h"
#include "conex/newton_step.h"

namespace conex {
using T = LinearKKTAssemblerBase;
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

void T::Increment(int* r, int sizer, int* c, int sizec,
                  Eigen::Map<MatrixXd>* data) {
  for (int i = 0; i < sizer; i++) {
    for (int j = 0; j < sizec; j++) {
      (*data)(i, j) += GetCoeff(*(r + i), *(c + j));
    }
  }
}

void T::Set(int* r, int sizer, int* c, int sizec, Eigen::Map<MatrixXd>* data) {
  for (int i = 0; i < sizer; i++) {
    for (int j = 0; j < sizec; j++) {
      (*data)(i, j) = GetCoeff(*(r + i), *(c + j));
    }
  }
}

double T::GetCoeff(int i, int j) {
  bool fill_in = (i < 0) || (j < 0);
  if (fill_in) {
    // Fill in is
    return 0;
  }
  return schur_complement_data.G(i, j);
}

void T::BindDiagonalBlock(const DiagonalBlock* data) {
  diag.push_back(*data);
  UpdateNumberOfVariables();
  int m = NumberOfVariables();

  schur_complement_data.m_ = m;
  memory.resize(SizeOf(schur_complement_data));
  Initialize(&schur_complement_data, memory.data());

  // TODO(FrankPermenter) Move this into smarter initialization function.
  if (m == data->num_vars) {
    new (&schur_complement_data.G)
        Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>(data->data, m, m);
  }
}

void T::BindOffDiagonalBlock(const OffDiagonalBlock* data) {
  if (data->stride != -1) {
    off_diag.push_back(*data);
  } else {
    scatter_block.push_back(*data);
  }

  UpdateNumberOfVariables();
  int m = NumberOfVariables();
  schur_complement_data.m_ = m;
  memory.resize(SizeOf(schur_complement_data));
  Initialize(&schur_complement_data, memory.data());
}

template <typename T>
vector<T> InitVector(T* data, int N) {
  vector<T> y;
  for (int i = 0; i < N; i++) {
    y.push_back(data[i]);
  }
  return y;
}

void T::Scatter(int* r, int sizer, int* c, int sizec, double** data) {
  int cnt = 0;
  for (size_t j = 0; j < sizec; j++) {
    for (size_t i = j; i < sizer; i++) {
      *data[cnt++] += GetCoeff(*(r + i), *(c + j));
    }
  }
}

void T::UpdateNumberOfVariables() {
  int max = 0;
  for (const auto& d : diag) {
    auto v = InitVector(d.var_data, d.num_vars);
    for (auto vi : v) {
      if (vi >= max) {
        max = vi + 1;
      }
    }
  }
  for (const auto& d : off_diag) {
    auto v = InitVector(d.row_data, d.num_rows);
    for (auto vi : v) {
      if (vi >= max) {
        max = vi + 1;
      }
    }
    v = InitVector(d.col_data, d.num_cols);
    for (auto vi : v) {
      if (vi >= max) {
        max = vi + 1;
      }
    }
  }
  for (const auto& d : scatter_block) {
    auto v = InitVector(d.row_data, d.num_rows);
    for (auto vi : v) {
      if (vi >= max) {
        max = vi + 1;
      }
    }
    v = InitVector(d.col_data, d.num_cols);
    for (auto vi : v) {
      if (vi >= max) {
        max = vi + 1;
      }
    }
  }
  num_variables_ = max;
}

void T::UpdateBlocks() {
  SetDenseData();

  for (const auto& d : diag) {
    // There is a single block and we have already filled it.
    if (d.num_vars == T::NumberOfVariables()) {
      return;
    }

    Eigen::Map<MatrixXd> data(d.data, d.num_vars, d.num_vars);
    // TODO(FrankPermenter): Only set the lower triangular part.
    if (d.assign) {
      Set(d.var_data, d.num_vars, d.var_data, d.num_vars, &data);
    } else {
      Increment(d.var_data, d.num_vars, d.var_data, d.num_vars, &data);
    }
  }

  for (const auto& d : off_diag) {
    Eigen::Map<MatrixXd> data(d.data, d.num_rows, d.num_cols);
    if (d.assign) {
      Set(d.row_data, d.num_rows, d.col_data, d.num_cols, &data);
    } else {
      Increment(d.row_data, d.num_rows, d.col_data, d.num_cols, &data);
    }
  }

  for (const auto& d : scatter_block) {
    Scatter(d.row_data, d.num_rows, d.col_data, d.num_cols, d.data_pointers);
  }
}

}  // namespace conex
