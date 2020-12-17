#pragma once
#include "newton_step.h"
#include <Eigen/Dense>

namespace conex {

struct WorkspaceSOC {
  WorkspaceSOC(int n) : n_(n), W(n + 1, 1) {}

  static constexpr int size_of(int n) {
    return get_size_aligned(n) + get_size_aligned(n + 1) +
           get_size_aligned(n + 1);
  }

  friend int SizeOf(const WorkspaceSOC& o) { return size_of(o.n_); }

  friend void Initialize(WorkspaceSOC* o, double* data) {
    using Map = Eigen::Map<DenseMatrix, Eigen::Aligned>;
    int n = o->n_;
    new (&o->W1) Map(data, n, 1);
    new (&o->temp_1) Map(data + get_size_aligned(n), n + 1, 1);
    new (&o->temp_2)
        Map(data + get_size_aligned(n) + get_size_aligned(n + 1), n + 1, 1);
  }

  friend void print(const WorkspaceSOC& o) {
    DUMP(o.W0);
    DUMP(o.W1);
    DUMP(o.temp_1);
    DUMP(o.temp_2);
  }

  double W0;
  Eigen::Map<DenseMatrix, Eigen::Aligned> W1{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> temp_1{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> temp_2{NULL, 0, 0};

  // Dummy to make dual variable interface happy
  int n_;
  Eigen::MatrixXd W;
};

}  // namespace conex
