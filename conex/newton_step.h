#pragma once
#include "debug_macros.h"
#include "memory_utils.h"
#include <Eigen/Dense>

namespace conex {

using DenseMatrix = Eigen::MatrixXd;
using Ref = Eigen::Map<DenseMatrix, Eigen::Aligned>;

struct WeightedSlackEigenvalues {
  double limit = 0;
  double frobenius_norm_squared = 0;
  double trace = 0;
  double lambda_min = std::numeric_limits<double>::max();
  double lambda_max = -std::numeric_limits<double>::max();
  double rank;
};

struct IterationStats {
  double norminf = 0;
};

struct StepOptions {
  bool affine = true;
  double inv_sqrt_mu = 0;
  // Take step of form  w_1 e + Q(w/2)(A^y - w_2 c)
  double c_weight = 0;
  double e_weight = 0;
  double step_size = 1;
};

struct StepInfo {
  double normsqrd = 0;
  double norminfd = 0;
};

using DenseMatrix = Eigen::MatrixXd;
struct WorkspaceSchurComplement {
  WorkspaceSchurComplement(int m) : m_(m) {}
  WorkspaceSchurComplement() {}

  static constexpr int size_of(int m, bool residual_only) {
    int size = 2 * get_size_aligned(m);
    if (!residual_only) {
      size += get_size_aligned(m * m);
    }
    return size;
  }

  friend int SizeOf(const WorkspaceSchurComplement& o) {
    return size_of(o.m_, o.residual_only_);
  }

  friend void Initialize(WorkspaceSchurComplement* o, double* data) {
    using Map = Eigen::Map<DenseMatrix, Eigen::Aligned>;
    int m = o->m_;
    new (&o->AW) Map(data, m, 1);
    new (&o->AQc) Map(data + 1 * get_size_aligned(m), m, 1);

    if (!o->residual_only_) {
      new (&o->G) Map(data + 2 * get_size_aligned(m), m, m);
    }

    o->initialized = true;
  }

  void setZero() {
    AW.setZero();
    AQc.setZero();
    inner_product_of_w_and_c = 0;
  }

  friend void print(const WorkspaceSchurComplement& o) {
    DUMP(o.initialized);
    DUMP(o.AW);
    DUMP(o.AQc);
  }

  double inner_product_of_w_and_c;

  Eigen::Map<DenseMatrix, Eigen::Aligned> G{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> AW{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> AQc{NULL, 0, 0};
  int m_;
  bool initialized = false;
  bool residual_only_ = false;
};

using SchurComplementSystem = WorkspaceSchurComplement;

}  // namespace conex
