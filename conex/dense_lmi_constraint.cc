#include "conex/dense_lmi_constraint.h"

/*
DenseLMIConstraint::DenseLMIConstraint(int n, std::vector<StorageType>* constraint_matrices,
                      StorageType* constraint_affine) : 
      constraint_matrices_(constraint_matrices), constraint_affine_(constraint_affine), 
      PsdConstraint(n, static_cast<int>(constraint_matrices->size()))  { }
      */


void DenseLMIConstraint::ComputeNegativeSlack(double k, const Ref& y, Ref* s) {
  MultByA(y, s);
  (*s) -= k * (constraint_affine_);
}

void DenseLMIConstraint::ComputeAW(int i, const Ref& W, Ref* AW, Ref* WAW) {
  auto& constraint_matrix = constraint_matrices_.at(i);
  AW->noalias() =  constraint_matrix * W;
  WAW->noalias() =  W * (*AW);
}

double DenseLMIConstraint::EvalDualConstraint(int j, const Ref& W) {
  const auto& constraint_matrix = constraint_matrices_.at(j);
  return (constraint_matrix * W).trace();
}

double DenseLMIConstraint::EvalDualObjective(const Ref& W) {
  const auto& constraint_matrix = constraint_affine_;
  return (constraint_matrix * W).trace();
}

void DenseLMIConstraint::MultByA(const Ref& x, Ref* Y) {
  const auto& constraint_matrices = constraint_matrices_;
  int i = 0;
  // TODO(FrankPermenter): Remove this setZero.
  Y->setZero();
  for (const auto& matrix : constraint_matrices) {
    (*Y) += x(i) * matrix;
    i++;
  }
}

template void ConstructSchurComplementSystem<DenseLMIConstraint>(
        DenseLMIConstraint* o, bool initialize, SchurComplementSystem* sys);
