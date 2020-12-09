#pragma once
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "conex/error_checking_macros.h"
#include "conex/newton_step.h"
#include "conex/workspace.h"

namespace conex {


template <typename T>
bool UpdateLinearOperator(T* obj, double val, int var, int row, int col,
                          int hyper_complex_dim) {
  CONEX_DEMAND(false,
               "Constraint does not support updates of linear operator.");
}

template <typename T>
bool UpdateAffineTerm(T*, double, int, int, int) {
  CONEX_DEMAND(false, "Constraint does not support updates of affine term.");
}

// A helper class for forwarding to different implementations of an "interface."
// With this approach, implementations do not need to use inheritance or virtual
// functions. Instead, they simply provide functions of appropriate name and
// signature, e.g.,
//
//    void TakeStep(Implementation1*, {arguments});
//    void Rank(Implementation1*, {arguments});
//    ..
//    void TakeStep(Implementation2*, {arguments});
//    void Rank(Implementation2*, {arguments});
//
// Note that implementations can be ANSI C compliant when the signature is.
//
// Reference: "Inheritance is the base-class of evil" by Sean Parent.  class
// Constraint {
class Constraint {
 public:
  template <typename Implementation>
  Constraint(const Implementation& t)
      : model(std::make_unique<Model<Implementation>>(t)) {}

  friend void ConstructSchurComplementSystem(Constraint* o, bool initialize,
                                             SchurComplementSystem* sys) {
    o->model->do_schur_complement(initialize, sys);
  }

  friend void SetIdentity(Constraint* o) { o->model->do_set_identity(); }

  friend void TakeStep(Constraint* o, const StepOptions& opt, const Ref& y,
                       StepInfo* info) {
    o->model->do_take_step(opt, y, info);
  }

  friend void GetMuSelectionParameters(Constraint* o, const Ref& y,
                                       MuSelectionParameters* p) {
    o->model->do_min_mu(y, p);
  }

  friend int Rank(const Constraint& o) { return o.model->do_rank(); }

  Workspace workspace() { return model->do_get_workspace(); }

  void get_dual_variable(double* v) { return model->do_get_dual_variable(v); }

  int dual_variable_size() { return model->do_dual_variable_size(); }

  int number_of_variables() { return model->do_number_of_variables(); }

  friend bool UpdateLinearOperator(Constraint* o, double val, int var, int row,
                                   int col, int hyper_complex_dim) {
    return o->model->do_update_linear_operator(val, var, row, col,
                                               hyper_complex_dim);
  }

  friend bool UpdateAffineTerm(Constraint* o, double val, int row, int col,
                               int hyper_complex_dim) {
    return o->model->do_update_affine_term(val, row, col, hyper_complex_dim);
  }

 private:
  struct Concept {
    virtual void do_schur_complement(bool initialize,
                                     SchurComplementSystem* sys) = 0;
    virtual void do_set_identity() = 0;
    virtual void do_min_mu(const Ref& y, MuSelectionParameters* p) = 0;
    virtual Workspace do_get_workspace() = 0;
    virtual void do_take_step(const StepOptions& opt, const Ref& y,
                              StepInfo* info) = 0;
    virtual void do_get_dual_variable(double*) = 0;
    virtual int do_dual_variable_size() = 0;
    virtual int do_number_of_variables() = 0;
    virtual bool do_update_linear_operator(double val, int var, int row,
                                           int col, int hyper_complex_dim) = 0;
    virtual bool do_update_affine_term(double val, int row, int col,
                                       int hyper_complex_dim) = 0;
    virtual int do_rank() = 0;
    virtual ~Concept() = default;
  };

  template <typename Implementation>
  struct Model final : Concept {
    Model(const Implementation& t) : data(t) {}

    void do_schur_complement(bool initialize,
                             SchurComplementSystem* sys) override {
      ConstructSchurComplementSystem(&data, initialize, sys);
    }

    void do_set_identity() override { SetIdentity(&data); }

    void do_min_mu(const Ref& y, MuSelectionParameters* p) override {
      GetMuSelectionParameters(&data, y, p);
    }

    int do_rank() override { return Rank(data); }

    int do_number_of_variables() override { return data.number_of_variables(); }

    Workspace do_get_workspace() override {
      return Workspace(data.workspace());
    }

    void do_get_dual_variable(double* var) override {
      memcpy(static_cast<void*>(var),
             static_cast<void*>(data.workspace()->W.data()),
             sizeof(double) * do_dual_variable_size());
    }

    int do_dual_variable_size() override {
      return data.workspace()->W.rows() * data.workspace()->W.cols();
    }

    bool do_update_linear_operator(double val, int var, int row, int col,
                                   int hyper_complex_dim) override {
      return UpdateLinearOperator(&data, val, var, row, col, hyper_complex_dim);
    }

    bool do_update_affine_term(double val, int row, int col,
                               int hyper_complex_dim) override {
      return UpdateAffineTerm(&data, val, row, col, hyper_complex_dim);
    }

    void do_take_step(const StepOptions& opt, const Ref& y,
                      StepInfo* info) override {
      TakeStep(&data, opt, y, info);
    }

    Implementation data;
  };
  std::unique_ptr<Concept> model;
};

} // namespace conex
