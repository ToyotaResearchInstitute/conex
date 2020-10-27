#pragma once
#include <Eigen/Dense>

#include "workspace.h"
#include "eigen_decomp.h"
#include "newton_step.h"

struct WorkspaceDensePSD {
 
  WorkspaceDensePSD(int n) : n_(n) {} 
  WorkspaceDensePSD(int n, double *data) :  
                                 W(data,  n, n),
                                 temp_1(data + get_size_aligned(n*n),  n, n),
                                 temp_2(data + get_size_aligned(n*n),  n, n)
  {}
  static constexpr int size_of(int n) { return 3*(get_size_aligned(n*n));  }

  friend int SizeOf(const WorkspaceDensePSD& o) {
    return size_of(o.n_);
  }

  friend void Initialize(WorkspaceDensePSD* o, double *data) {
    using Map = Eigen::Map<DenseMatrix, Eigen::Aligned>;
     int n = o->n_;
     new (&o->W)  Map(data, n, n);
     new (&o->temp_1) Map(data + 1*get_size_aligned(n*n),  n, n);
     new (&o->temp_2) Map(data + 2*get_size_aligned(n*n),  n, n);
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


// To avoid overhead of virtual functions, we 
// assume that classes that inherit PsdConstraint
// will add a specialization of this template
// as a "friend" function.
template<typename T>
void ConstructSchurComplementSystem(T* o, 
                                bool initialize,
                                SchurComplementSystem* sys) {
auto G = &sys->G;

auto workspace = o->workspace();
auto& W = workspace->W; auto& AW = workspace->temp_1;
auto& WAW = workspace->temp_2;
int m = o->num_dual_constraints_;  
if (initialize) {
  for (int i = 0; i < m; i++) {
    o->ComputeAW(i, W, &AW, &WAW);
    for (int j = i; j < m; j++) {
      (*G)(j, i) = o->EvalDualConstraint(j, WAW);
    }

    sys->AW(i, 0)  = AW.trace();
    sys->AQc(i, 0) = o->EvalDualObjective(WAW);
  }
} else {
  for (int i = 0; i < m; i++) {
    o->ComputeAW(i, W, &AW, &WAW);
    for (int j = i; j < m; j++) {
      (*G)(j, i) += o->EvalDualConstraint(j, WAW);
    }

    sys->AW(i, 0)   += AW.trace();
    sys->AQc(i, 0)  += o->EvalDualObjective(WAW);
  }
}
}

class PsdConstraint {
 public:
  friend void SetIdentity(PsdConstraint* o);
  friend int Rank(const PsdConstraint& o) { return o.workspace_.n_; };
  WorkspaceDensePSD* workspace() { return &workspace_; }
  friend void TakeStep(PsdConstraint* o, const StepOptions& opt, const Ref& y, StepInfo*);
  friend void ComputeStats(PsdConstraint* o, const StepOptions& opt, const Ref& y);
  friend void GetMuSelectionParameters(PsdConstraint* o,  const Ref& y, MuSelectionParameters* p);

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

};
