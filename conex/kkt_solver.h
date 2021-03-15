#pragma once
#include "constraint_manager.h"

#include "conex/RLDLT.h"
#include "conex/supernodal_solver.h"

namespace conex {

class Solver {
 public:
  Solver(const std::vector<std::vector<int>>& cliques,
         const std::vector<std::vector<int>>& dual_vars);

  Solver(const std::vector<std::vector<int>>& cliques,
          int num_vars, std::vector<int>& order, 
          const std::vector<std::vector<int>>& supernodes,
          const std::vector<std::vector<int>>& separators);



  void Bind(const std::vector<KKT_SystemAssembler*>& kkt_assembler);

  void Bind(std::vector<KKT_SystemAssembler>* kkt_assembler) {
    std::vector<KKT_SystemAssembler*> pointers;
    for (auto& e : *kkt_assembler) {
      pointers.push_back(&e);
    }
    Bind(pointers);
  }

  void RelabelCliques(MatrixData* data_ptr);
  void Assemble(Eigen::VectorXd* AW, Eigen::VectorXd* AWc,
                double* inner_product_of_c_and_w);
  bool Factor();
  Eigen::VectorXd Solve(const Eigen::VectorXd& b);
  void SolveInPlace(Eigen::Map<Eigen::MatrixXd, Eigen::Aligned>* b);
  Eigen::MatrixXd KKTMatrix();

 private:
  bool use_cholesky_ = false;
  // Copies of inputs.
  const std::vector<std::vector<int>> cliques_;
  const std::vector<std::vector<int>> dual_variables_;
  MatrixData data;
  SparseTriangularMatrix mat;
  std::vector<Eigen::RLDLT<Eigen::Ref<Eigen::MatrixXd>>> factorization;
  Eigen::PermutationMatrix<-1> Pt;
  std::vector<KKT_SystemAssembler*> assembler;
};

}  // namespace conex
