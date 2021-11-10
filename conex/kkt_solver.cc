#include "conex/kkt_solver.h"

#include "conex/block_triangular_operations.h"
#include "conex/kkt_system_assembler.h"
#include "conex/supernodal_solver.h"

using Eigen::VectorXd;
using std::vector;
namespace {

std::vector<int> ConcatFirstN(const std::vector<int>& a, int N,
                              const std::vector<int>& b) {
  std::vector<int> y;
  for (int i = 0; i < N; i++) {
    y.push_back(a.at(i));
  }
  for (size_t j = 0; j < b.size(); j++) {
    y.push_back(b.at(j));
  }
  return y;
}

std::vector<int> ReplaceWithPosition(const std::vector<int>& a,
                                     const std::vector<int>& b,
                                     bool label_fill_in) {
  std::vector<int> y;
  for (auto ai : a) {
    auto p = std::find(b.begin(), b.end(), ai);
    if (p != b.end()) {
      y.push_back(std::distance(b.begin(), p));
    } else {
      if (label_fill_in) {
        y.push_back(-1);
      } else {
        assert(0);
      }
    }
  }
  return y;
}

}  // namespace

namespace conex {

using T = SupernodalKKTSolver;
void T::RelabelCliques(MatrixData* data_ptr) {
  auto cliques = cliques_;
  auto& data = *data_ptr;

  // Replace original variables with their position in the
  // constraint clique.
  for (int e = 0; e < static_cast<int>(cliques.size()); e++) {
    // Here we assume that dual variables are at end of clique.
    int j = data.clique_order.at(e);
    auto labels = ConcatFirstN(
        cliques_.at(j), cliques_.at(j).size() - dual_variables_.at(j).size(),
        dual_variables_.at(j));

    data.supernodes_original_labels.at(e) =
        ReplaceWithPosition(data.supernodes_original_labels.at(e), labels,
                            /*!found = fill in*/ true);

    data.separators_original_labels.at(e) =
        ReplaceWithPosition(data.separators_original_labels.at(e), labels,
                            /*!found = fill in*/ true);
  }
}

int GetRootNode(const std::vector<std::vector<int>>& vars,
                const std::vector<std::vector<int>>& dual_vars) {
  int arg_max = 0;

  size_t max = dual_vars.at(0).size();
  for (size_t i = 1; i < dual_vars.size(); i++) {
    if (dual_vars.at(i).size() > max) {
      arg_max = i;
      max = dual_vars.at(i).size();
    }
  }
  if (max > 0) {
    return arg_max;
  }

  arg_max = 0;
  max = vars.at(0).size();
  for (size_t i = 1; i < vars.size(); i++) {
    if (vars.at(i).size() > max) {
      arg_max = i;
      max = vars.at(i).size();
    }
  }
  return arg_max;
}

vector<int> is_empty(const vector<std::vector<int>>& vect) {
  vector<int> y(vect.size());
  for (size_t i = 0; i < y.size(); i++) {
    y[i] = vect[i].size() == 0;
  }
  return y;
}

T::SupernodalKKTSolver(const std::vector<std::vector<int>>& cliques,
                       const std::vector<std::vector<int>>& dual_vars)
    : cliques_(cliques),
      dual_variables_(dual_vars),
      data(GetData(cliques, is_empty(dual_vars),
                   GetRootNode(cliques, dual_vars))),
      mat(data),
      Pt(data.N),
      b_permuted_(data.N) {
  RelabelCliques(&data);
  Pt.indices() =
      Eigen::Map<Eigen::MatrixXi>(data.permutation_inverse.data(), data.N, 1);
}

using std::vector;

T::SupernodalKKTSolver(const std::vector<std::vector<int>>& cliques,
                       int num_vars, const std::vector<int>& order,
                       const std::vector<std::vector<int>>& supernodes,
                       const std::vector<std::vector<int>>& separators)
    : cliques_(cliques),
      dual_variables_(vector<vector<int>>(cliques.size())),
      data(SupernodesToData(num_vars, order, supernodes, separators)),
      mat(data),
      Pt(data.N),
      b_permuted_(data.N) {
  RelabelCliques(&data);
  Pt.indices() =
      Eigen::Map<Eigen::MatrixXi>(data.permutation_inverse.data(), data.N, 1);
}

void T::Assemble(Eigen::VectorXd* AW, Eigen::VectorXd* AQc,
                 double* inner_product_of_c_and_w) {
  if (AW->rows() != SizeOfSystem() || AQc->rows() != SizeOfSystem()) {
    throw std::runtime_error(
        "Cannot assemble system data: invalid output dimensions.");
  }
  const auto& cliques = cliques_;

  for (int e = static_cast<int>(cliques.size()) - 1; e >= 0; e--) {
    int i = data.clique_order.at(e);
    assembler.at(i)->UpdateBlocks();
  }

  if (AW && AQc && inner_product_of_c_and_w) {
    AW->setZero();
    AQc->setZero();
    *inner_product_of_c_and_w = 0;
    for (int e = static_cast<int>(cliques.size()) - 1; e >= 0; e--) {
      int i = data.clique_order.at(e);
      auto* rhs_i = assembler.at(i)->GetWorkspace();
      *inner_product_of_c_and_w += rhs_i->inner_product_of_w_and_c;
      int cnt = 0;
      for (auto k : cliques.at(i)) {
        (*AW)(k) += rhs_i->AW(cnt);
        (*AQc)(k) += rhs_i->AQc(cnt);
        cnt++;
      }
    }
  }
}

void T::Assemble() {
  const auto& cliques = cliques_;
  for (int e = static_cast<int>(cliques.size()) - 1; e >= 0; e--) {
    int i = data.clique_order.at(e);
    assembler.at(i)->UpdateBlocks();
  }
}

bool T::Factor() {
  // TODO(FrankPermenter): save a sparse copy of the matrix instead.
  bool use_qr = mode_ == CONEX_QR_FACTORIZATION;
  if (iterative_refinement_iterations_ > 0 || use_qr) {
    kkt_matrix_ = KKTMatrix();
  }

  if (!use_qr) {
    use_cholesky_ = true;
    for (auto di : dual_variables_) {
      if (di.size() > 0) {
        use_cholesky_ = false;
        break;
      }
    }
    if (use_cholesky_) {
      return BlockTriangularOperations::BlockCholeskyInPlace(&mat.workspace_);
    } else {
      bool no_regularization = BlockTriangularOperations::BlockLDLTInPlace(
          &mat.workspace_, &factorization);
      factorization_regularized_ = !no_regularization;
      return true;
    }
  } else {
    qr_decomp_.compute(kkt_matrix_);
    return true;
  }
}

Eigen::VectorXd T::Solve(const Eigen::VectorXd& b) const {
  assert(b.rows() == Pt.rows());

  bool use_qr = mode_ == CONEX_QR_FACTORIZATION;
  if (!use_qr) {
    b_permuted_ = Pt.transpose() * b;
    if (use_cholesky_) {
      BlockTriangularOperations::SolveInPlaceCholesky(mat.workspace_,
                                                      &b_permuted_);
    } else {
      BlockTriangularOperations::SolveInPlaceLDLT(mat.workspace_, factorization,
                                                  &b_permuted_);
    }
    return Pt * b_permuted_;
  } else {
    return qr_decomp_.solve(b);
  }
}

double T::SolveInPlace(Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>* b) const {
  bool use_qr = mode_ == CONEX_QR_FACTORIZATION;
  double residual_norm = -1;
  if (b->rows() != Pt.rows()) {
    throw std::runtime_error(
        "Supernodal solver input error: invalid dimensions.");
  }

  if (use_qr) {
    VectorXd sol = qr_decomp_.solve(*b);
    *b = sol;
    return -1;
  }

  Eigen::VectorXd total_residual;
  if (iterative_refinement_iterations_ > 0) {
    total_residual = *b;
  }
  b_permuted_ = Pt.transpose() * (*b);

  if (use_cholesky_) {
    BlockTriangularOperations::SolveInPlaceCholesky(mat.workspace_,
                                                    &b_permuted_);
  } else {
    BlockTriangularOperations::SolveInPlaceLDLT(mat.workspace_, factorization,
                                                &b_permuted_);
  }
  *b = Pt * b_permuted_;
  for (int i = 0; i < iterative_refinement_iterations_; i++) {
    auto& y = *b;
    const VectorXd residual = total_residual - kkt_matrix_ * y;
    residual_norm = residual.norm();
    b_permuted_ = Pt.transpose() * (residual);
    if (use_cholesky_) {
      BlockTriangularOperations::SolveInPlaceCholesky(mat.workspace_,
                                                      &b_permuted_);
    } else {
      BlockTriangularOperations::SolveInPlaceLDLT(mat.workspace_, factorization,
                                                  &b_permuted_);
    }
    VectorXd temp = Pt * b_permuted_;
    y += temp;
  }
  return residual_norm;
}

Eigen::MatrixXd T::KKTMatrix() const {
  Eigen::MatrixXd G =
      TriangularMatrixOperations::ToDense(mat).selfadjointView<Eigen::Lower>();
  return Pt * G * Pt.transpose();
}

}  // namespace conex
