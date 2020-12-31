#pragma once
#include <Eigen/Dense>

#include "newton_step.h"
#include "workspace.h"

namespace conex {

struct WorkspaceDensePSD {
  WorkspaceDensePSD(int n) : n_(n) {}
  WorkspaceDensePSD(int n, double* data)
      : W(data, n, n),
        temp_1(data + get_size_aligned(n * n), n, n),
        temp_2(data + get_size_aligned(n * n), n, n) {}
  static constexpr int size_of(int n) { return 3 * (get_size_aligned(n * n)); }

  friend int SizeOf(const WorkspaceDensePSD& o) { return size_of(o.n_); }

  friend void Initialize(WorkspaceDensePSD* o, double* data) {
    using Map = Eigen::Map<DenseMatrix, Eigen::Aligned>;
    int n = o->n_;
    new (&o->W) Map(data, n, n);
    new (&o->temp_1) Map(data + 1 * get_size_aligned(n * n), n, n);
    new (&o->temp_2) Map(data + 2 * get_size_aligned(n * n), n, n);
  }

  friend void print(const WorkspaceDensePSD& o) {
    DUMP(o.W);
    DUMP(o.temp_1);
    DUMP(o.temp_2);
  }

  Eigen::Map<DenseMatrix, Eigen::Aligned> W{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> temp_1{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> temp_2{NULL, 0, 0};
  int n_;
};

class PsdConstraint {
 public:
  friend void SetIdentity(PsdConstraint* o);
  friend int Rank(const PsdConstraint& o) { return o.workspace_.n_; };
  WorkspaceDensePSD* workspace() { return &workspace_; }
  friend void TakeStep(PsdConstraint* o, const StepOptions& opt, const Ref& y,
                       StepInfo*);
  friend void GetMuSelectionParameters(PsdConstraint* o, const Ref& y,
                                       MuSelectionParameters* p);
  int number_of_variables() { return num_dual_constraints_; }

 protected:
  PsdConstraint(int n, int m) : workspace_(n), num_dual_constraints_{m} {}
  void GeodesicUpdate(double scale, const StepOptions&, Ref* sw);
  void AffineUpdate(double e_weight, Ref* sw);

  WorkspaceDensePSD workspace_;
  int num_dual_constraints_;
  virtual double EvalDualConstraint(int j, const Ref& W) = 0;
  virtual double EvalDualObjective(const Ref& W) = 0;
  virtual void ComputeAW(int i, const Ref& W, Ref* AW, Ref* WAW) = 0;
  virtual void ComputeNegativeSlack(double k, const Ref& y, Ref* s) = 0;
  virtual ~PsdConstraint(){};
};

}  // namespace conex
