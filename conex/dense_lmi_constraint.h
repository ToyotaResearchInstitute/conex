#include "psd_constraint.h"
#include "newton_step.h"

//using InputRefType = Eigen::Map<const DenseMatrix>;
using InputRefType = DenseMatrix;

struct DenseLMIConstraint final : public PsdConstraint {
  DenseLMIConstraint(int n, const std::vector<DenseMatrix>& constraint_matrices,
                     const DenseMatrix& constraint_affine) : 
      PsdConstraint(n, static_cast<int>(constraint_matrices.size())),
      constraint_matrices_(constraint_matrices), constraint_affine_(constraint_affine) {}

  const std::vector<DenseMatrix> constraint_matrices_;
  const DenseMatrix constraint_affine_;

  friend void ConstructSchurComplementSystem<DenseLMIConstraint>(
        DenseLMIConstraint* o, bool initialize, SchurComplementSystem* sys);

 private:
  void ComputeNegativeSlack(double k, const Ref& y, Ref* s);
  void ComputeAW(int i, const Ref& W, Ref* AW, Ref* WAW);
  double EvalDualConstraint(int j, const Ref& W);
  double EvalDualObjective(const Ref& W);
  void MultByA(const Ref& x, Ref* Y);
};
