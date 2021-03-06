#pragma once
#include "debug_macros.h"
#include "memory_utils.h"
#include <Eigen/Dense>

namespace conex {

using DenseMatrix = Eigen::MatrixXd;
using Ref = Eigen::Map<DenseMatrix, Eigen::Aligned>;

struct MuSelectionParameters {
  double limit = 0;
  double gw_norm_squared = 0;
  double gw_trace = 0;
  double gw_lambda_min = 1e30;
  double gw_lambda_max = -1e30;
};

struct IterationStats {
  double norminf = 0;
};

struct StepOptions {
  bool affine = true;
  double inv_sqrt_mu = 0;
  double stepsize = 0;
  // Take step of form  w_1 e + Q(w/2)(A^y - w_2 c)
  double c_weight = 0;
  double e_weight = 0;
};

struct StepInfo {
  double normsqrd;
  double norminfd;
};

using DenseMatrix = Eigen::MatrixXd;
struct WorkspaceSchurComplement {
  WorkspaceSchurComplement(int m) : m_(m) {}
  WorkspaceSchurComplement() {}

  static constexpr int size_of(int m) {
    return get_size_aligned(m * m) + 3 * get_size_aligned(m);
  }

  friend int SizeOf(const WorkspaceSchurComplement& o) { return size_of(o.m_); }

  friend void Initialize(WorkspaceSchurComplement* o, double* data) {
    using Map = Eigen::Map<DenseMatrix, Eigen::Aligned>;
    int m = o->m_;
    new (&o->G) Map(data, m, m);
    // TODO(FrankPermenter): Remove b.
    new (&o->b) Map(data + get_size_aligned(m * m), m, 1);
    new (&o->AW)
        Map(data + get_size_aligned(m * m) + get_size_aligned(m), m, 1);
    new (&o->AQc)
        Map(data + get_size_aligned(m * m) + 2 * get_size_aligned(m), m, 1);
    o->initialized = true;
  }

  friend void print(const WorkspaceSchurComplement& o) {
    DUMP(o.initialized);
    DUMP(o.G);
    DUMP(o.b);
    DUMP(o.AW);
    DUMP(o.AQc);
  }

  double QwCNorm;
  double QwCTrace;
  Eigen::Map<DenseMatrix, Eigen::Aligned> G{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> b{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> AW{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> AQc{NULL, 0, 0};
  int m_;
  bool initialized = false;
};

using SchurComplementSystem = WorkspaceSchurComplement;

}  // namespace conex
