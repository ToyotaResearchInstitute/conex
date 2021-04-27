#pragma once
#include <numeric>
#include <vector>

#include <Eigen/Dense>

#include "conex/debug_macros.h"
#include "conex/memory_utils.h"

namespace conex {

using Clique = std::vector<int>;

struct TriangularMatrixWorkspace {
  TriangularMatrixWorkspace(const std::vector<Clique>& path_,
                            const std::vector<int>& supernode_size_);
  int N;
  // TODO(FrankPermenter): Remove all of these members.
  std::vector<int> supernode_size;
  std::vector<Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>> diagonal;
  std::vector<Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>> off_diagonal;
  std::vector<std::vector<double*>> seperator_diagonal;

  std::vector<std::vector<int>> snodes;
  std::vector<std::vector<int>> separators;

  int SizeOfSupernode(int i) const {
    return get_size_aligned(supernode_size.at(i) * supernode_size.at(i));
  }

  int SizeOfSeparator(int i) const {
    return get_size_aligned(supernode_size.at(i) * separators.at(i).size());
  }

  friend int SizeOf(const TriangularMatrixWorkspace& o) {
    int size = 0;
    for (size_t j = 0; j < o.snodes.size(); j++) {
      size += o.SizeOfSupernode(j);
    }
    for (size_t j = 0; j < o.separators.size(); j++) {
      size += o.SizeOfSeparator(j);
    }
    return size;
  }

  friend void Initialize(TriangularMatrixWorkspace* o, double* data_start);

  // A cache of IntersectionOfSupernodeAndSeparator. The first records
  // the j for which IntersectionOfSupernodeAndSeparator(i+1, j) is nonempty.
  // The second returns the output.
  std::vector<std::vector<int>> column_intersections;
  std::vector<std::vector<std::vector<std::pair<int, int>>>>
      intersection_position;

  std::vector<Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>>> llts;

  // Needed for solving linear systems.
  mutable std::vector<Eigen::VectorXd> temporaries;

  std::vector<int> variable_to_supernode_;
  std::vector<int> variable_to_supernode_position_;

 private:
  // TODO(FrankPermenter): Remove this method.
  void S_S(int clique, std::vector<double*>*);
};

}  // namespace conex
