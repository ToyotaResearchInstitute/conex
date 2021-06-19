#include "conex/kkt_solver.h"

#include "conex/block_triangular_operations.h"
#include "conex/kkt_system_assembler.h"
#include "conex/supernodal_solver.h"

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

using T = Solver;
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

T::Solver(const std::vector<std::vector<int>>& cliques,
          const std::vector<std::vector<int>>& dual_vars)
    : cliques_(cliques),
      dual_variables_(dual_vars),
      data(GetData(cliques, GetRootNode(cliques, dual_vars))),
      mat(data),
      Pt(data.N),
      b_permuted_(data.N) {
  RelabelCliques(&data);
  Pt.indices() =
      Eigen::Map<Eigen::MatrixXi>(data.permutation_inverse.data(), data.N, 1);
}

using std::vector;

T::Solver(const std::vector<std::vector<int>>& cliques, int num_vars,
          const std::vector<int>& order,
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
    return BlockTriangularOperations::BlockLDLTInPlace(&mat.workspace_,
                                                       &factorization);
  }
}

// TODO(FrankPermenter): Reimplement Solve using SolveInPlace.
Eigen::VectorXd T::Solve(const Eigen::VectorXd& b) {
  assert(b.rows() == Pt.rows());
  b_permuted_ = Pt.transpose() * b;

  if (use_cholesky_) {
    BlockTriangularOperations::SolveInPlaceCholesky(mat.workspace_,
                                                    &b_permuted_);
  } else {
    BlockTriangularOperations::SolveInPlaceLDLT(mat.workspace_, factorization,
                                                &b_permuted_);
  }

  return Pt * b_permuted_;
}

void T::SolveInPlace(Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>* b) {
  if (b->rows() != Pt.rows()) {
    throw std::runtime_error(
        "Supernodal solver input error: invalid dimensions.");
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
}

Eigen::MatrixXd T::KKTMatrix() {
  Eigen::MatrixXd G =
      TriangularMatrixOperations::ToDense(mat).selfadjointView<Eigen::Lower>();
  return Pt * G * Pt.transpose();
}

}  // namespace conex
