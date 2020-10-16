#pragma once
#include <vector>
#include <memory>

#include <Eigen/Dense>

#include "workspace.h"
#include "newton_step.h"


class Constraint {
 public:
  template <typename T>
  Constraint(const T& t) : model(std::make_unique<Model<T>>(t)) { }

  friend void ConstructSchurComplementSystem(Constraint* o, bool initialize, SchurComplementSystem* sys) {
    o->model->do_schur_complement(initialize, sys);
  }

  friend void SetIdentity(Constraint* o) {
    o->model->do_set_identity();
  }

  friend void TakeStep(Constraint* o, const StepOptions& opt, const Ref& y, StepInfo* info) {
    o->model->do_take_step(opt, y, info);
  }

  friend void MinMu(Constraint* o, const Ref& y, MuSelectionParameters* p) {
    o->model->do_min_mu(y, p);
  }

  friend int Rank(const Constraint& o) {
    return o.model->do_rank();
  }

  Workspace workspace() {
    return model->do_get_workspace();
  }

  void get_dual_variable(double* v) {
    return model->do_get_dual_variable(v);
  }

  int dual_variable_size() {
    return model->do_dual_variable_size();
  }

 private:
  struct Concept {
    virtual void do_schur_complement(bool initialize, SchurComplementSystem* sys) = 0;
    virtual void do_set_identity() = 0;
    virtual void do_min_mu(const Ref& y, MuSelectionParameters* p) = 0;
    virtual Workspace do_get_workspace() = 0;
    virtual void do_take_step(const StepOptions& opt, const Ref& y, StepInfo* info) = 0;
    virtual void do_get_dual_variable(double*) = 0;
    virtual int do_dual_variable_size() = 0;
    virtual int do_rank() = 0;
    virtual ~Concept()  = default;
  };

  template <typename T>
  struct Model final : Concept {
    Model(const T& t) : data(t) {}

    void do_schur_complement(bool initialize, SchurComplementSystem* sys) override {
      ConstructSchurComplementSystem(&data, initialize, sys);
    }

    void do_set_identity() override {
      SetIdentity(&data);
    }

    void do_min_mu(const Ref& y, MuSelectionParameters* p) override {
      MinMu(&data, y, p);
    }

    int do_rank() override {
      return Rank(data);
    }

    Workspace do_get_workspace() override {
      return Workspace(data.workspace());
    }

    void do_get_dual_variable(double *var) override {
      memcpy(static_cast<void *>(var), 
             static_cast<void *>(data.workspace()->W.data()),
             sizeof(double) * do_dual_variable_size());
      
    }

    int do_dual_variable_size() override {
       return data.workspace()->W.rows() * data.workspace()->W.cols();
    }

    void do_take_step(const StepOptions& opt, const Ref& y, StepInfo* info) override {
      TakeStep(&data, opt, y, info); 
    }

    T data;
  };
  std::unique_ptr<Concept> model;
};

