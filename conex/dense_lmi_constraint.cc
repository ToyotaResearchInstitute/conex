#include "conex/dense_lmi_constraint.h"


void MatrixLMIConstraint::ComputeNegativeSlack(double k, const Ref& y, Ref* s) {
  MultByA(y, s);
  (*s) -= k * (constraint_affine_);
}

void MatrixLMIConstraint::ComputeAW(int i, const Ref& W, Ref* AW, Ref* WAW) {
  auto& constraint_matrix = constraint_matrices_.at(i);
  AW->noalias() =  constraint_matrix * W;
  WAW->noalias() =  W * (*AW);
}

double MatrixLMIConstraint::EvalDualConstraint(int j, const Ref& W) {
  const auto& constraint_matrix = constraint_matrices_.at(j);
  return (constraint_matrix * W).trace();
}

double MatrixLMIConstraint::EvalDualObjective(const Ref& W) {
  const auto& constraint_matrix = constraint_affine_;
  return (constraint_matrix * W).trace();
}

void MatrixLMIConstraint::MultByA(const Ref& x, Ref* Y) {
  const auto& constraint_matrices = constraint_matrices_;
  int i = 0;
  Y->setZero();
  for (const auto& matrix : constraint_matrices) {
    (*Y) += x(variable(i)) * matrix;
    i++;
  }
}

template void ConstructSchurComplementSystem<DenseLMIConstraint>(
        DenseLMIConstraint* o, bool initialize, SchurComplementSystem* sys);

template void ConstructSchurComplementSystem<SparseLMIConstraint>(
        SparseLMIConstraint* o, bool initialize, SchurComplementSystem* sys);
