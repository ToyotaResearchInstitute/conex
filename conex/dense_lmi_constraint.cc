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

double TraceInnnerProduct(const Eigen::MatrixXd& X, const Ref& Y) {
  double val = 0;
  for (int i = 0; i < X.rows(); i++) {
    val += X.col(i).dot(Y.col(i));
  }
  return val;
}

double MatrixLMIConstraint::EvalDualConstraint(int j, const Ref& W) {
  const auto& constraint_matrix = constraint_matrices_.at(j);
  return TraceInnnerProduct(constraint_matrix, W);
}

double MatrixLMIConstraint::EvalDualObjective(const Ref& W) {
  const auto& constraint_matrix = constraint_affine_;
  return TraceInnnerProduct(constraint_matrix, W);
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
