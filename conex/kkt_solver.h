#pragma once
#include "constraint_manager.h"

#include "conex/RLDLT.h"
#include "conex/supernodal_assembler.h"
#include "conex/supernodal_solver.h"

namespace conex {

enum : int {
  CONEX_LLT_FACTORIZATION = 0,
  CONEX_LDLT_FACTORIZATION = 1,
  CONEX_QR_FACTORIZATION = 2,
};

class SupernodalKKTSolver {
 public:
  SupernodalKKTSolver(const std::vector<std::vector<int>>& cliques,
                      const std::vector<std::vector<int>>& dual_vars);

  SupernodalKKTSolver(const std::vector<std::vector<int>>& cliques,
                      int num_vars, const std::vector<int>& order,
                      const std::vector<std::vector<int>>& supernodes,
                      const std::vector<std::vector<int>>& separators);

  template <typename SupernodalAssemblerDerivedClass>
  void Bind(const std::vector<SupernodalAssemblerDerivedClass*>&
                supernodal_assembler) {
    DoBind(data, mat.workspace_, supernodal_assembler);
    for (auto v : supernodal_assembler) {
      assembler.push_back(v);
    }
  }

  void Assemble(Eigen::VectorXd* AW, Eigen::VectorXd* AWc,
                double* inner_product_of_c_and_w);
  void Assemble();
  void SetIterativeRefinementIterations(int x) {
    iterative_refinement_iterations_ = x;
  }
  void SetSolverMode(int mode) { mode_ = mode; }
  bool Factor();
  Eigen::VectorXd Solve(const Eigen::VectorXd& b) const;
  void SolveInPlace(Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>* b) const;
  Eigen::MatrixXd KKTMatrix() const;

 private:
  void RelabelCliques(MatrixData* data_ptr);
  int SizeOfSystem() { return Pt.rows(); }
  bool use_cholesky_ = false;
  // Copies of inputs.
  const std::vector<std::vector<int>> cliques_;
  const std::vector<std::vector<int>> dual_variables_;
  MatrixData data;
  SparseTriangularMatrix mat;
  std::vector<Eigen::RLDLT<Eigen::Ref<Eigen::MatrixXd>>> factorization;
  Eigen::PermutationMatrix<-1> Pt;
  mutable Eigen::VectorXd b_permuted_;
  std::vector<SupernodalAssemblerBase*> assembler;
  bool factorization_regularized_ = false;
  int iterative_refinement_iterations_ = 0;
  mutable Eigen::MatrixXd kkt_matrix_;
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_decomp_;
  int mode_ = CONEX_LLT_FACTORIZATION;
};

}  // namespace conex
