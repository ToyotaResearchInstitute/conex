#pragma once
#include <vector>
#include "conex/linear_workspace.h"
#include "conex/supernodal_cholesky_data.h"

#include "conex/constraint.h"
#include "conex/newton_step.h"
#include <Eigen/Dense>

namespace conex {

// For a given clique, manages the transfer of
// densely stored data to supernodal data structure.
class LinearKKTAssemblerBase {
 public:
  void BindDiagonalBlock(const DiagonalBlock* data);
  void BindOffDiagonalBlock(const OffDiagonalBlock* data);

  void UpdateBlocks();
  virtual void SetDenseData() = 0;

  int NumberOfVariables() { return num_variables_; };
  void UpdateNumberOfVariables();

  WorkspaceSchurComplement schur_complement_data;
  Eigen::VectorXd memory;

 protected:
  double GetCoeff(int i, int j);
  void Increment(std::vector<int> r, std::vector<int> c,
                 Eigen::Map<Eigen::MatrixXd>* data);

  void Increment(int* r, int sizer, int* c, int sizec,
                 Eigen::Map<Eigen::MatrixXd>* data);
  void Set(int* r, int sizer, int* c, int sizec,
           Eigen::Map<Eigen::MatrixXd>* data);

  void Set(std::vector<int> r, std::vector<int> c,
           Eigen::Map<Eigen::MatrixXd>* data);

  void SetDiagonalBlock(std::vector<int> r, Eigen::Map<Eigen::MatrixXd>* data);
  void Scatter(int* r, int sizer, int* c, int sizec, double** data);
  int num_variables_;

  std::vector<DiagonalBlock> diag;
  std::vector<OffDiagonalBlock> off_diag;
  std::vector<OffDiagonalBlock> scatter_block;
};

class LinearKKTAssembler : public LinearKKTAssemblerBase {
 public:
  LinearKKTAssembler(int num_variables, Constraint* W) {
    workspace_ = W;
    num_variables_ = num_variables;
    assert(W);
  }
  virtual void SetDenseData() {
    if (workspace_) {
      ConstructSchurComplementSystem(workspace_, true, &schur_complement_data);
    }
  }

  LinearKKTAssembler(){};
  Constraint* workspace_ = NULL;
};

class LinearKKTAssemblerStatic : public LinearKKTAssemblerBase {
 public:
  LinearKKTAssemblerStatic(){};
  LinearKKTAssemblerStatic(const Eigen::MatrixXd& A) : A_(A) {}

  virtual void SetDenseData() override {
    schur_complement_data.G = A_.transpose() * A_;
  }
  Eigen::MatrixXd A_;
};

}  // namespace conex
