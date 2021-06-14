#pragma once
#include "conex/constraint.h"
#include "conex/constraint_manager.h"
#include "conex/equality_constraint.h"
#include "conex/error_checking_macros.h"
#include "conex/kkt_assembler.h"
#include "conex/kkt_solver.h"
#include "workspace.h"

namespace conex {

//  Allocates memory for internal computations.

struct SolverConfiguration {
  int prepare_dual_variables = 0;
  int initialization_mode = 0;
  // TODO(FrankPermenter): Remove inv_sqrt_mu_max
  double inv_sqrt_mu_max = 1000;
  double minimum_mu = 1e-12;
  double maximum_mu = 1e4;
  double divergence_upper_bound = 1;
  int final_centering_steps = 5;
  double final_centering_tolerance = .01;
  int initial_centering_steps_warmstart = 0;
  int initial_centering_steps_coldstart = 0;
  double warmstart_abort_threshold = 2;
  int max_iterations = 25;
  double infeasibility_threshold = 1e5;
};

struct ConexStatus {
  int solved = 0;
  int primal_infeasible = 0;
  int dual_infeasible = 0;
};

class Container {
 public:
  template <typename T>
  Container(const T& x, int num_vars)
      : obj(x), constraint(std::any_cast<T>(&obj)) {
    kkt_assembler.SetNumberOfVariables(num_vars);
  }
  std::any obj;
  Constraint constraint;
  LinearKKTAssembler kkt_assembler;
};

class Program {
 public:
  Program(int number_of_variables) {
    workspace_data_ = &memory_;
    SetNumberOfVariables(number_of_variables);
  }

  Program(int number_of_variables, Eigen::VectorXd* data) {
    workspace_data_ = data;
    SetNumberOfVariables(number_of_variables);
  }

  void SetNumberOfVariables(int m) {
    kkt_system_manager_.SetNumberOfVariables(m);
    sys.m_ = m;
  }

  int GetNumberOfVariables() { return sys.m_; }

  template <typename T>
  void GetDualVariable(int i, T* xi) {
    int cnt = 0;
    for (auto& ci : kkt_system_manager_.eqs) {
      if (cnt == i) {
        ci.constraint.get_dual_variable(xi->data());
        if (status_.solved) {
          xi->array() /=
              (stats->sqrt_inv_mu[stats->num_iter - 1] * stats->b_scaling());
        }
        return;
      }
      cnt++;
    }
  }

  int GetDualVariableSize(int i) {
    int cnt = 0;
    for (auto& ci : kkt_system_manager_.eqs) {
      if (cnt == i) {
        return ci.constraint.dual_variable_size();
      }
      cnt++;
    }
    CONEX_DEMAND(false, "Invalid Constraint");
  }

  int UpdateLinearOperatorOfConstraint(int i, double value, int variable,
                                       int row, int col,
                                       int hyper_complex_dim) {
    int cnt = 0;
    for (auto& ci : kkt_system_manager_.eqs) {
      if (cnt == i) {
        return UpdateLinearOperator(&ci.constraint, value, variable, row, col,
                                    hyper_complex_dim);
      }
      cnt++;
    }
    CONEX_DEMAND(false, "Invalid Constraint");
  }

  int UpdateAffineTermOfConstraint(int i, double value, int row, int col,
                                   int hyper_complex_dim) {
    int cnt = 0;
    for (auto& ci : kkt_system_manager_.eqs) {
      if (cnt == i) {
        return UpdateAffineTerm(&ci.constraint, value, row, col,
                                hyper_complex_dim);
      }
      cnt++;
    }
    CONEX_DEMAND(false, "Invalid Constraint");
  }

  void InitializeWorkspace() {
    workspaces.clear();
    for (auto& c : kkt_system_manager_.eqs) {
      workspaces.push_back(c.constraint.workspace());
      workspaces.emplace_back(&c.kkt_assembler.schur_complement_data);
    }
    workspaces.emplace_back(stats.get());
    workspaces.emplace_back(&sys);
    auto size = SizeOf(workspaces);
    if (size > workspace_data_->size()) {
      workspace_data_->resize(size);
    }
    Initialize(&workspaces, workspace_data_->data());

    is_initialized = true;
  }

  template <typename T>
  bool AddConstraint(T&& d) {
    if constexpr (!std::is_same<T, EqualityConstraints>::value) {
      bool result = kkt_system_manager_.AddConstraint<T>(std::forward<T>(d));
      if (result) {
        constraints.push_back(&kkt_system_manager_.eqs.back().constraint);
      }
      return result;
    } else {
      return kkt_system_manager_.AddEqualityConstraint(
          std::forward<EqualityConstraints>(d));
    }
  }

  template <typename T>
  bool AddConstraint(T&& d, const std::vector<int>& variables) {
    if constexpr (!std::is_same<T, EqualityConstraints>::value) {
      bool result =
          kkt_system_manager_.AddConstraint<T>(std::forward<T>(d), variables);
      if (result) {
        constraints.push_back(&kkt_system_manager_.eqs.back().constraint);
      }
      return result;
    } else {
      return kkt_system_manager_.AddEqualityConstraint(
          std::forward<EqualityConstraints>(d), variables);
    }
  }

  int NumberOfConstraints() { return kkt_system_manager_.eqs.size(); }
  ConexStatus Status() { return status_; }

  ConstraintManager<Container> kkt_system_manager_;
  std::vector<Constraint*> constraints;
  SchurComplementSystem sys;
  std::unique_ptr<WorkspaceStats> stats;
  std::vector<Workspace> workspaces;
  std::unique_ptr<Solver> solver;
  std::vector<KKT_SystemAssembler> kkt;
  Eigen::VectorXd memory_;
  Eigen::VectorXd* workspace_data_;
  bool is_initialized = false;
  ConexStatus status_;
};

DenseMatrix GetFeasibleObjective(Program* prog);
bool Solve(const DenseMatrix& b, Program& prog,
           const SolverConfiguration& config, double* primal_variable);

}  // namespace conex
