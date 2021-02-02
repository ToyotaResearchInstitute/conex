#pragma once
#include "newton_step.h"
#include <Eigen/Dense>

namespace conex {

struct WorkspaceSOC {
  WorkspaceSOC(int n) : n_(n), W(n + 1, 1) {}

  static constexpr int size_of(int n) {
    return get_size_aligned(n) + 4 * get_size_aligned(n);
  }

  friend int SizeOf(const WorkspaceSOC& o) { return size_of(o.n_); }

  friend void Initialize(WorkspaceSOC* o, double* data) {
    using Map = Eigen::Map<DenseMatrix, Eigen::Aligned>;
    int n = o->n_;
    new (&o->W1) Map(data, n, 1);
    new (&o->temp1_1) Map(data + get_size_aligned(n), n, 1);
    new (&o->temp2_1) Map(data + 2 * get_size_aligned(n), n, 1);
    new (&o->temp3_1) Map(data + 3 * get_size_aligned(n), n, 1);
    o->W0 = data + 4 * get_size_aligned(n);
  }

  friend void print(const WorkspaceSOC& o) {
    DUMP(o.W0);
    DUMP(o.W1);
    DUMP(o.temp1_1);
    DUMP(o.temp2_1);
    DUMP(o.temp3_1);
  }

  double* W0;
  double d0;
  double wsqrt_q1_norm_sqr;
  // TODO(FrankPermenter): Reduce number of temporaries.
  Eigen::Map<DenseMatrix, Eigen::Aligned> W1{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> temp1_1{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> temp2_1{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> temp3_1{NULL, 0, 0};

  // Dummy to make dual variable interface happy
  // TODO(FrankPermenter): Remove this.
  int n_;
  Eigen::MatrixXd W;
};

}  // namespace conex
