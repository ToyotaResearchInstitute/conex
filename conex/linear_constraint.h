#pragma once
#include "conex/constraint.h"
#include "conex/newton_step.h"
#include "linear_workspace.h"

namespace conex {

void PreprocessLinearInequality(const Eigen::MatrixXd& A,
                                const Eigen::MatrixXd& lb,
                                const Eigen::MatrixXd& ub,
                                Eigen::MatrixXd* Aineq, Eigen::MatrixXd* bineq,
                                Eigen::MatrixXd* Aeq, Eigen::MatrixXd* beq);
// TODO(FrankPermenter) Rename to LinearInequality
class LinearConstraint {
  using StorageType = DenseMatrix;

 public:
  template <typename T>
  LinearConstraint(int n, T* constraint_matrix, T* constraint_affine)
      : LinearConstraint(n, *constraint_matrix, *constraint_affine) {}

  template <typename T>
  LinearConstraint(const T& constraint_matrix, const T& constraint_affine)
      : LinearConstraint(constraint_matrix.rows(), constraint_matrix,
                         constraint_affine) {}

  LinearConstraint(const Eigen::MatrixXd& constraint_matrix,
                   const Eigen::MatrixXd& constraint_affine)
      : LinearConstraint(constraint_matrix.rows(), constraint_matrix,
                         constraint_affine) {}

  template <typename T>
  LinearConstraint(int n, const T& constraint_matrix,
                   const T& constraint_affine)
      : workspace_(n, constraint_matrix.cols()),
        constraint_matrix_(constraint_matrix),
        constraint_affine_(constraint_affine) {
    assert(constraint_affine_.rows() == n);
    assert(constraint_matrix_.rows() == n);
  }

  LinearConstraint(int n, int m, const double* constraint_matrix,
                   const double* constraint_affine)
      : LinearConstraint(
            n, Eigen::Map<const DenseMatrix>(constraint_matrix, n, m),
            Eigen::Map<const DenseMatrix>(constraint_affine, n, 1)) {}

  WorkspaceLinear* workspace() { return &workspace_; }

  int number_of_variables() { return constraint_matrix_.cols(); }
  friend int Rank(const LinearConstraint& o) { return o.workspace_.n_; };
  friend void SetIdentity(LinearConstraint* o);
  friend void PrepareStep(LinearConstraint* o, const StepOptions& opt,
                          const Ref& y, StepInfo* data);

  friend bool PerformLineSearch(LinearConstraint* o,
                                const LineSearchParameters& params,
                                const Ref& y0, const Ref& y1,
                                LineSearchOutput* output);

  // Eigenvalues of Q(w^{1/2}) *(c-A*y)
  friend void GetWeightedSlackEigenvalues(LinearConstraint* o, const Ref& y,
                                          double c_weight,
                                          WeightedSlackEigenvalues* p);

  friend void ConstructSchurComplementSystem(LinearConstraint* o,
                                             bool initialize,
                                             SchurComplementSystem* sys);
  friend bool TakeStep(LinearConstraint*, const StepOptions&);

  friend bool UpdateLinearOperator(LinearConstraint* o, double val, int var,
                                   int r, int c, int dim);
  friend bool UpdateAffineTerm(LinearConstraint* o, double val, int r, int c,
                               int dim);

 private:
  void ComputeNegativeSlack(double inv_sqrt_mu, const Ref& y, Ref* minus_s);
  void GeodesicUpdate(const Ref& S, StepInfo* data);
  void AffineUpdate(const Ref& S);

  WorkspaceLinear workspace_;
  DenseMatrix constraint_matrix_;
  DenseMatrix constraint_affine_;
};

class LowerBound : public LinearConstraint {
 public:
  LowerBound(const Eigen::VectorXd& lower_bounds)
      : LinearConstraint(-Eigen::MatrixXd::Identity(lower_bounds.rows(),
                                                    lower_bounds.rows()),
                         -lower_bounds) {}

  friend bool PerformLineSearch(LowerBound* o,
                                const LineSearchParameters& params,
                                const Ref& y0, const Ref& y1,
                                LineSearchOutput* output) {
    return PerformLineSearch(static_cast<LinearConstraint*>(o), params, y0, y1,
                             output);
  }
};

class UpperBound : public LinearConstraint {
 public:
  UpperBound(const Eigen::VectorXd& upper_bounds)
      : LinearConstraint(
            Eigen::MatrixXd::Identity(upper_bounds.rows(), upper_bounds.rows()),
            upper_bounds) {}

  friend bool PerformLineSearch(UpperBound* o,
                                const LineSearchParameters& params,
                                const Ref& y0, const Ref& y1,
                                LineSearchOutput* output) {
    return PerformLineSearch(static_cast<LinearConstraint*>(o), params, y0, y1,
                             output);
  }
};

}  // namespace conex
