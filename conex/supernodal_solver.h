#pragma once
#include <numeric>
#include <vector>

#include <Eigen/Dense>

#include "conex/supernodal_assembler.h"
#include "conex/triangular_matrix_workspace.h"

namespace conex {

using Clique = std::vector<int>;

std::vector<int> UnionOfSorted(const std::vector<int>& x1,
                               const std::vector<int>& x2);

// TODO(FrankPermenter): Deprecate this struct.
struct MatrixData {
  std::vector<std::vector<int>> cliques;
  std::vector<std::vector<int>> supernodes_original_labels;
  std::vector<std::vector<int>> separators_original_labels;
  std::vector<int> supernode_size;
  std::vector<int> permutation;
  // The map from supernode to original variable.
  std::vector<int> permutation_inverse;
  // The map from clique (with fill-in) to original clique.
  std::vector<int> clique_order;
  int N;
};

MatrixData GetData(const std::vector<Clique>& cliques, int init = 0);

MatrixData GetData(const std::vector<Clique>& cliques,
                   const std::vector<int>& valid_leafs, int init = 0);

template <typename T>
inline void DoBind(const MatrixData& data, TriangularMatrixWorkspace& workspace,
                   const std::vector<T*>& eqs) {
  auto& sn = data.supernodes_original_labels;
  auto& sep = data.separators_original_labels;

  for (int e = static_cast<int>(eqs.size()) - 1; e >= 0; e--) {
    int i = data.clique_order.at(e);

    if (sn.at(e).size() > 0) {
      auto blockD = BuildBlock(&sn.at(e), workspace.diagonal.at(e).data());
      eqs.at(i)->BindDiagonalBlock(&blockD);
    }

    if (sep.at(e).size() > 0 && sn.at(e).size() > 0) {
      auto block = BuildBlock(&sn.at(e), &sep.at(e),
                              workspace.off_diagonal.at(e).data());
      eqs.at(i)->BindOffDiagonalBlock(&block);
    }

    if (workspace.seperator_diagonal.at(e).size() > 0) {
      auto block = BuildBlock(&sep.at(e), &sep.at(e),
                              &workspace.seperator_diagonal.at(e));
      eqs.at(i)->BindOffDiagonalBlock(&block);
    }
  }
}

class SparseTriangularMatrix {
 public:
  SparseTriangularMatrix(int num_cols, const std::vector<Clique>& cliques,
                         const std::vector<int>& supernode_sizes,
                         const Eigen::VectorXd& memory)
      : N(num_cols),
        workspace_(cliques, supernode_sizes),
        memory_(memory),
        path(cliques),
        supernode_size(workspace_.supernode_size),
        supernodes(workspace_.diagonal),
        snodes(workspace_.snodes),
        separator(workspace_.off_diagonal) {
    assert(memory_.size() >= SizeOf(workspace_));
    Initialize(&workspace_, memory_.data());
  }

  SparseTriangularMatrix(int N_, const std::vector<Clique>& cliques,
                         const std::vector<int>& supernode_sizes)
      : SparseTriangularMatrix(
            N_, cliques, supernode_sizes,
            Eigen::VectorXd::Zero(
                SizeOf(TriangularMatrixWorkspace(cliques, supernode_sizes)))) {}

  SparseTriangularMatrix(const MatrixData& data)
      : SparseTriangularMatrix(data.N, data.cliques, data.supernode_size) {}

  SparseTriangularMatrix(const SparseTriangularMatrix& s)
      : SparseTriangularMatrix(s.N, s.path, s.supernode_size, s.memory_) {}

  SparseTriangularMatrix operator=(const SparseTriangularMatrix& s) {
    return SparseTriangularMatrix(s.N, s.path, s.supernode_size, s.memory_);
  }

  int N;
  TriangularMatrixWorkspace workspace_;
  Eigen::VectorXd memory_;
  std::vector<Clique> path;
  std::vector<int>& supernode_size;
  std::vector<Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>>& supernodes;
  std::vector<std::vector<int>>& snodes;
  std::vector<Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>>& separator;
};
std::vector<Clique> Permute(std::vector<Clique>& path,
                            std::vector<int>& permutation);
void Sort(std::vector<Clique>* path);

void IntersectionOfSorted(const std::vector<int>& v1,
                          const std::vector<int>& v2, std::vector<int>* v3);

class TriangularMatrixOperations {
 public:
  using Matrix = SparseTriangularMatrix;
  static Eigen::MatrixXd Multiply(SparseTriangularMatrix& mat,
                                  const Eigen::MatrixXd& x);
  static void SetConstant(SparseTriangularMatrix* mat, double val);
  static Eigen::MatrixXd ToDense(const SparseTriangularMatrix& mat);
  static void RescaleColumn(Matrix* mat);
  static void CholeskyInPlace(SparseTriangularMatrix* mat);
  static Eigen::VectorXd ApplyInverse(SparseTriangularMatrix* L,
                                      const Eigen::VectorXd& b);
  static Eigen::VectorXd ApplyInverseOfTranspose(SparseTriangularMatrix* L,
                                                 const Eigen::VectorXd& b);
};

MatrixData SupernodesToData(int num_vars, const std::vector<int>& order,
                            const std::vector<std::vector<int>>& supernodes,
                            const std::vector<std::vector<int>>& separators);

}  // namespace conex
