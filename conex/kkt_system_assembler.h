#pragma once

#include <memory>
#include <vector>
#include "conex/newton_step.h"
#include "conex/supernodal_cholesky_data.h"

#include <Eigen/Dense>

namespace conex {

class KKT_SystemAssembler {
 public:
  template <typename Implementation>
  KKT_SystemAssembler(Implementation* t)
      : model(std::make_unique<Dispatcher<Implementation>>(t)) {}

  void BindDiagonalBlock(const DiagonalBlock* data) {
    model->BindDiagonalBlock(data);
  }

  void BindOffDiagonalBlock(const OffDiagonalBlock* data) {
    model->BindOffDiagonalBlock(data);
  }

  void UpdateBlocks() { model->UpdateBlocks(); }
  void Reset() { model->Reset(); }

  SchurComplementSystem* GetWorkspace() { return model->GetWorkspace(); }

 private:
  class KKT_SystemAssemblerDispatcher {
   public:
    virtual void BindDiagonalBlock(const DiagonalBlock* data) = 0;
    virtual void BindOffDiagonalBlock(const OffDiagonalBlock* data) = 0;
    virtual void UpdateBlocks() = 0;
    virtual void Reset() = 0;
    virtual SchurComplementSystem* GetWorkspace() = 0;
    virtual ~KKT_SystemAssemblerDispatcher(){};
  };

  template <typename Implementation>
  struct Dispatcher final : KKT_SystemAssemblerDispatcher {
    Dispatcher(Implementation* t) : implementation(t) {}

    void BindDiagonalBlock(const DiagonalBlock* data) override {
      implementation->BindDiagonalBlock(data);
    }
    void BindOffDiagonalBlock(const OffDiagonalBlock* data) override {
      implementation->BindOffDiagonalBlock(data);
    }
    void UpdateBlocks() override { implementation->UpdateBlocks(); }
    void Reset() override { implementation->Reset(); }

    WorkspaceSchurComplement* GetWorkspace() override {
      return &implementation->schur_complement_data;
    }

    Implementation* implementation;
  };

  std::unique_ptr<KKT_SystemAssemblerDispatcher> model;
};

inline OffDiagonalBlock BuildBlock(std::vector<int>* r, std::vector<int>* c,
                                   double* matrix_data) {
  OffDiagonalBlock block;
  block.num_rows = r->size();
  block.row_data = r->data();
  block.num_cols = c->size();
  block.col_data = c->data();
  block.stride = 1;
  block.data = matrix_data;
  block.assign = 1;
  return block;
}

inline DiagonalBlock BuildBlock(std::vector<int>* r, double* matrix_data) {
  DiagonalBlock block;
  block.num_vars = r->size();
  block.var_data = r->data();
  block.stride = 1;
  block.data = matrix_data;
  block.assign = 1;
  return block;
}

inline OffDiagonalBlock BuildBlock(std::vector<int>* r, std::vector<int>* c,
                                   std::vector<double*>* mat) {
  OffDiagonalBlock block;
  block.num_rows = r->size();
  block.row_data = r->data();
  block.num_cols = c->size();
  block.col_data = c->data();
  block.stride = -1;
  block.data_pointers = mat->data();
  block.assign = 0;
  return block;
}

}  // namespace conex
