#pragma once
#include "constraint_manager.h"

#include "conex/RLDLT.h"
#include "conex/supernodal_solver.h"

namespace conex {

enum : int {
  CONEX_LLT_FACTORIZATION = 0,
  CONEX_LDLT_FACTORIZATION = 1,
  CONEX_QR_FACTORIZATION = 2,
};

class Solver {
 public:
  Solver(const std::vector<std::vector<int>>& cliques,
         const std::vector<std::vector<int>>& dual_vars);

  Solver(const std::vector<std::vector<int>>& cliques, int num_vars,
         const std::vector<int>& order,
         const std::vector<std::vector<int>>& supernodes,
         const std::vector<std::vector<int>>& separators);

  template <typename T>
  void Bind(const std::vector<T*>& kkt_assembler) {
    DoBind(data, mat.workspace_, kkt_assembler);
  }

  template <typename T>
  void AssembleFromCliques(const std::vector<T*>& assemblers) {
    const auto& cliques = cliques_;
    for (int e = static_cast<int>(cliques.size()) - 1; e >= 0; e--) {
      int i = data.clique_order.at(e);
      assemblers.at(i)->UpdateBlocks();
    }
  }

  void Bind(std::vector<KKT_SystemAssembler>* kkt_assembler) {
    std::vector<KKT_SystemAssembler*> pointers;
    for (auto& e : *kkt_assembler) {
      pointers.push_back(&e);
    }
    Bind(pointers);
  }

  void Bind(const std::vector<KKT_SystemAssembler*>& kkt_assembler) {
    DoBind(data, mat.workspace_, kkt_assembler);
    assembler = kkt_assembler;
  }

  void RelabelCliques(MatrixData* data_ptr);
  void Assemble(Eigen::VectorXd* AW, Eigen::VectorXd* AWc,
                double* inner_product_of_c_and_w);
  void Assemble();
  void SetIterativeRefinementIterations(int x) {
    iterative_refinement_iterations_ = x;
  }
  void SetSolverMode(int mode) { mode_ = mode; }

  bool Factor();
  Eigen::VectorXd Solve(const Eigen::VectorXd& b);
  double SolveInPlace(Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>* b);
  Eigen::MatrixXd KKTMatrix();

 private:
  int SizeOfSystem() { return Pt.rows(); }
  bool use_cholesky_ = false;
  // Copies of inputs.
  const std::vector<std::vector<int>> cliques_;
  const std::vector<std::vector<int>> dual_variables_;
  MatrixData data;
  SparseTriangularMatrix mat;
  std::vector<Eigen::RLDLT<Eigen::Ref<Eigen::MatrixXd>>> factorization;
  Eigen::PermutationMatrix<-1> Pt;
  Eigen::VectorXd b_permuted_;
  std::vector<KKT_SystemAssembler*> assembler;
  bool factorization_regularized_ = false;
  int iterative_refinement_iterations_ = 0;
  Eigen::MatrixXd kkt_matrix_;
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_decomp_;
  int mode_;
};

}  // namespace conex
