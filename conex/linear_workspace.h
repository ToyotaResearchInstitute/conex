#pragma once
#include "conex/debug_macros.h"
#include "conex/memory_utils.h"
#include <Eigen/Dense>
namespace conex {

struct WorkspaceLinear {
  using DenseMatrix = Eigen::MatrixXd;
  WorkspaceLinear(int n, int num_vars) : n_(n), num_vars_(num_vars) {}

  static constexpr int size_of(int n, int num_vars) {
    return 3 * (get_size_aligned(n)) + get_size_aligned(n * num_vars);
  }

  friend int SizeOf(const WorkspaceLinear& o) {
    return size_of(o.n_, o.num_vars_);
  }

  friend void Initialize(WorkspaceLinear* o, double* data) {
    using Map = Eigen::Map<DenseMatrix, Eigen::Aligned>;
    int n = o->n_;
    new (&o->W) Map(data, n, 1);
    new (&o->temp_1) Map(data + 1 * get_size_aligned(n), n, 1);
    new (&o->temp_2) Map(data + 2 * get_size_aligned(n), n, 1);
    new (&o->weighted_constraints)
        Map(data + 3 * get_size_aligned(n), n, o->num_vars_);
  }

  friend void print(const WorkspaceLinear& o) {
    DUMP(o.W);
    DUMP(o.temp_1);
    DUMP(o.temp_2);
  }

  Eigen::Map<DenseMatrix, Eigen::Aligned> W{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> temp_1{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> temp_2{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> weighted_constraints{NULL, 0, 0};
  int n_;
  int num_vars_;
};

}  // namespace conex
// namespace conex
