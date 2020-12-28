#pragma once
#include <vector>

namespace conex {

struct DiagonalBlock {
  int num_vars;
  const int* var_data;
  double* data;
  int increment_or_assign;
  int stride;
  int assign;
};

struct OffDiagonalBlock {
  int num_rows;
  const int* row_data;
  int num_cols;
  const int* col_data;
  double* data;
  double** data_pointers;
  int assign;
  int stride;
};

}  // namespace conex
