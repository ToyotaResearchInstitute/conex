#include "conex/newton_step.h"
#include "conex/workspace_soc.h"

namespace conex {

class SOCConstraint {
  using StorageType = DenseMatrix;

 public:
  template <typename T>
  SOCConstraint(const T& constraint_matrix, const T& constraint_affine)
      : workspace_(constraint_matrix.rows() - 1),
        constraint_matrix_(constraint_matrix),
        constraint_affine_(constraint_affine) {
    assert(constraint_matrix_.rows() == constraint_affine_.rows());
  }

  // Lorentz cone a subset of R^(n+1).
  SOCConstraint(int n) : workspace_(n), n_(n) {}

  WorkspaceSOC* workspace() { return &workspace_; }

  int number_of_variables() { return constraint_matrix_.cols(); }
  friend int Rank(const SOCConstraint&) { return 2; };
  friend void SetIdentity(SOCConstraint* o) {
    *o->workspace_.W0 = 1;
    o->workspace_.W1.setZero();
  }
  friend void PrepareStep(SOCConstraint* o, const StepOptions& opt,
                          const Ref& y, StepInfo* data);

  friend bool TakeStep(SOCConstraint* o, const StepOptions& opt);

  friend void GetWeightedSlackEigenvalues(SOCConstraint* o, const Ref& y,
                                          double c_weight,
                                          WeightedSlackEigenvalues* p);
  friend void ConstructSchurComplementSystem(SOCConstraint* o, bool initialize,
                                             SchurComplementSystem* sys);

  friend bool UpdateLinearOperator(SOCConstraint* o, double val, int var, int r,
                                   int c, int dim);
  friend bool UpdateAffineTerm(SOCConstraint* o, double val, int r, int c,
                               int dim);

 private:
  void ComputeNegativeSlack(double inv_sqrt_mu, const Ref& y, Ref* minus_s);
  void GeodesicUpdate(const Ref& S, StepInfo* data);
  void AffineUpdate(const Ref& S);

  WorkspaceSOC workspace_;
  DenseMatrix constraint_matrix_;
  DenseMatrix constraint_affine_;
  int n_;
};

}  // namespace conex
