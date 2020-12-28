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

std::vector<int> Relabel(const std::vector<int>& x,
                         const std::vector<int>& labels) {
  std::vector<int> y;
  for (auto& xi : x) {
    y.push_back(labels.at(xi));
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

  std::vector<int> order;
  std::vector<std::vector<int>> separators;
  std::vector<std::vector<int>> supernodes;

  data.supernodes_original_labels.resize(cliques.size());
  data.separators_original_labels.resize(cliques.size());
  // Match to original variables.
  for (int e = 0; e < static_cast<int>(cliques.size()); e++) {
    data.supernodes_original_labels.at(e) =
        Relabel(data.supernodes.at(e), data.permutation_inverse);
    data.separators_original_labels.at(e) =
        Relabel(data.separators.at(e), data.permutation_inverse);
  }

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
                            /*!found = error*/ false);

    data.separators_original_labels.at(e) =
        ReplaceWithPosition(data.separators_original_labels.at(e), labels,
                            /*!found = fill in*/ true);
  }
}

T::Solver(const std::vector<std::vector<int>>& cliques,
          const std::vector<std::vector<int>>& dual_vars)
    : cliques_(cliques),
      dual_variables_(dual_vars),
      data(GetData(cliques)),
      mat(data),
      Pt(data.N) {
  RelabelCliques(&data);
  Pt.indices() =
      Eigen::Map<Eigen::MatrixXi>(data.permutation_inverse.data(), data.N, 1);
}

void T::Bind(const std::vector<KKT_SystemAssembler*>& kkt_assembler) {
  DoBind(data, mat.workspace_, kkt_assembler);
  assembler = kkt_assembler;
}

void T::Assemble(Eigen::VectorXd* AW, Eigen::VectorXd* AQc) {
  const auto& cliques = cliques_;
  AW->setZero();
  AQc->setZero();
  for (int e = static_cast<int>(cliques.size()) - 1; e >= 0; e--) {
    int i = data.clique_order.at(e);
    assembler.at(i)->UpdateBlocks();
    auto* rhs_i = assembler.at(i)->GetWorkspace();
    int cnt = 0;
    for (auto k : cliques.at(i)) {
      (*AW)(k) += rhs_i->AW(cnt);
      (*AQc)(k) += rhs_i->AQc(cnt);
      cnt++;
    }
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
  Eigen::VectorXd y = Pt.transpose() * b;

  if (use_cholesky_) {
    BlockTriangularOperations::SolveInPlaceCholesky(mat.workspace_, &y);
  } else {
    BlockTriangularOperations::SolveInPlaceLDLT(mat.workspace_, factorization,
                                                &y);
  }

  return Pt * y;
}

void T::SolveInPlace(Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>* b) {
  auto temp = T::Solve(*b);
  *b = temp;
}

Eigen::MatrixXd T::KKTMatrix() {
  Eigen::MatrixXd G =
      TriangularMatrixOperations::ToDense(mat).selfadjointView<Eigen::Lower>();
  return Pt * G * Pt.transpose();
}

}  // namespace conex
