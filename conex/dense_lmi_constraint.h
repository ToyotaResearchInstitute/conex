#pragma once
#include "newton_step.h"
#include "psd_constraint.h"

namespace conex {

class MatrixLMIConstraint : public PsdConstraint {
 public:
  MatrixLMIConstraint(int n,
                      const std::vector<DenseMatrix>& constraint_matrices,
                      const DenseMatrix& constraint_affine);

  Eigen::MatrixXd constraint_matrices_vect_;
  const std::vector<DenseMatrix> constraint_matrices_;
  const DenseMatrix constraint_affine_;

 protected:
  void ComputeAW(int i, const Ref& W, Ref* AW, Ref* WAW);
  void ComputeWCW(const Ref& W, Ref* CW, Ref* WCW);
  double EvalDualConstraint(int j, const Ref& W);
  double EvalDualObjective(const Ref& W);
};

class DenseLMIConstraint final : public MatrixLMIConstraint {
 public:
  DenseLMIConstraint(int n, const std::vector<DenseMatrix>& constraint_matrices,
                     const DenseMatrix& constraint_affine)
      : MatrixLMIConstraint(n, constraint_matrices, constraint_affine) {}

  DenseLMIConstraint(const std::vector<DenseMatrix>& constraint_matrices,
                     const DenseMatrix& constraint_affine)
      : MatrixLMIConstraint(constraint_affine.rows(), constraint_matrices,
                            constraint_affine) {}

  friend void ConstructSchurComplementSystem(DenseLMIConstraint* o,
                                             bool initialize,
                                             SchurComplementSystem* sys);

 private:
  void ComputeNegativeSlack(double k, const Ref& y, Ref* s) override;
};
}  // namespace conex
