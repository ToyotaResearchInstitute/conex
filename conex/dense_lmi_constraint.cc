#include "conex/dense_lmi_constraint.h"

namespace conex {

namespace {
using Eigen::MatrixXd;

template <bool sparse>
void MultByA(const Ref& x, Ref* Y, std::vector<MatrixXd> constraint_matrices,
             std::vector<int> variable = {}) {
  int i = 0;
  Y->setZero();
  for (const auto& matrix : constraint_matrices) {
    if constexpr (sparse) {
      (*Y) += x(variable.at(i)) * matrix;
    } else {
      (*Y) += x(i) * matrix;
    }
    i++;
  }
}
}  // namespace

void DenseLMIConstraint::ComputeNegativeSlack(double k, const Ref& y, Ref* s) {
  MultByA<false>(y, s, constraint_matrices_);
  (*s) -= k * (constraint_affine_);
}

void MatrixLMIConstraint::ComputeAW(int i, const Ref& W, Ref* AW, Ref* WAW) {
  auto& constraint_matrix = constraint_matrices_.at(i);
  AW->noalias() = constraint_matrix * W;
  WAW->noalias() = W * (*AW);
}

void MatrixLMIConstraint::ComputeWCW(const Ref& W, Ref* CW, Ref* WCW) {
  auto& constraint_matrix = constraint_affine_;
  CW->noalias() = constraint_matrix * W;
  WCW->noalias() = W * (*CW);
}
MatrixLMIConstraint::MatrixLMIConstraint(
    int n, const std::vector<DenseMatrix>& constraint_matrices,
    const DenseMatrix& constraint_affine)
    : PsdConstraint(n, static_cast<int>(constraint_matrices.size())),
      constraint_matrices_(constraint_matrices),
      constraint_affine_(constraint_affine) {
  int m = constraint_matrices_.size();
  constraint_matrices_vect_.resize(n * n, m);
  for (int i = 0; i < m; i++) {
    memcpy(&(constraint_matrices_vect_(0, i)),
           constraint_matrices_.at(i).data(), sizeof(double) * n * n);
  }
}

double TraceInnerProduct(const Eigen::MatrixXd& X, const Ref& Y) {
  double val = 0;
  for (int i = 0; i < X.rows(); i++) {
    val += X.col(i).dot(Y.col(i));
  }
  return val;
}

double MatrixLMIConstraint::EvalDualConstraint(int j, const Ref& W) {
  const auto& constraint_matrix = constraint_matrices_.at(j);
  return TraceInnerProduct(constraint_matrix, W);
}

double MatrixLMIConstraint::EvalDualObjective(const Ref& W) {
  const auto& constraint_matrix = constraint_affine_;
  return TraceInnerProduct(constraint_matrix, W);
}

#define SCHUR_COMPLEMENT_FUNCTION(OP)                                        \
  int n = Rank(*o);                                                          \
  Eigen::Map<Eigen::VectorXd> vectWAW(WAW.data(), n* n);                     \
  for (int i = 0; i < m; i++) {                                              \
    o->ComputeAW(i, W, &AW, &WAW);                                           \
    sys->G.row(i).head(i + 1) OP vectWAW.transpose() *                       \
        o->constraint_matrices_vect_.leftCols(i + 1);                        \
    sys->AW(i, 0) OP AW.trace();                                             \
    sys->AQc(i, 0) OP o->EvalDualObjective(WAW);                             \
  }                                                                          \
  sys->inner_product_of_w_and_c OP o->EvalDualObjective(W);                  \
                                                                             \
  auto& WCW = WAW;                                                           \
  auto& CW = AW;                                                             \
  o->ComputeWCW(W, &CW, &WCW);                                               \
  sys->inner_product_of_c_and_Qc OP TraceInnerProduct(o->constraint_affine_, \
                                                      WCW);

void ConstructSchurComplementSystem(DenseLMIConstraint* o, bool initialize,
                                    SchurComplementSystem* sys) {
  auto workspace = o->workspace();
  auto& W = workspace->W;
  auto& AW = workspace->temp_1;
  auto& WAW = workspace->temp_2;
  int m = o->num_dual_constraints_;

  if (initialize) {
    SCHUR_COMPLEMENT_FUNCTION(=);
  } else {
    SCHUR_COMPLEMENT_FUNCTION(+=);
  }
}

}  // namespace conex
