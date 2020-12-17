#pragma once
#include "conex/constraint_manager.h"
#include "conex/kkt_assembler.h"
#include "conex/kkt_solver.h"
#include "constraint.h"
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
  int max_iterations = 25;
  double infeasibility_threshold = 1e5;
};

class Container {
 public:
  template <typename T>
  Container(const T& x) : obj(x), constraint(std::any_cast<T>(&obj)) {}
  std::any obj;
  Constraint constraint;
  LinearKKTAssembler kkt_assembler;
};

class Program {
 public:
  Program(int number_of_variables) {
    SetNumberOfVariables(number_of_variables);
  }

  void SetNumberOfVariables(int m) {
    kkt_system_manager_.SetNumberOfVariables(m);
    sys.m_ = m;
  }

  int GetNumberOfVariables() { return sys.m_; }

  int SetNumberOfConstraints() const { return kkt_system_manager_.eqs.size(); }

  void InitializeWorkspace() {
    for (auto& constraint : kkt_system_manager_.eqs) {
      workspaces.push_back(constraint.constraint.workspace());
    }
    workspaces.push_back(Workspace{&stats});
    workspaces.push_back(Workspace{&sys});
    memory.resize(SizeOf(workspaces));
    Initialize(&workspaces, &memory[0]);

    constraints.clear();
    for (auto& ci : kkt_system_manager_.eqs) {
      constraints.push_back(&ci.constraint);
    }

    is_initialized = true;
  }

  template <typename T>
  void AddConstraint(T&& d) {
    kkt_system_manager_.AddConstraint(d);
  }

  template <typename T>
  void AddConstraint(T&& d, const std::vector<int>& variables) {
    kkt_system_manager_.AddConstraint(d, variables);
  }

  int NumberOfConstraints() { return kkt_system_manager_.eqs.size(); }

  ConstraintManager<Container> kkt_system_manager_;
  std::vector<Constraint*> constraints;
  SchurComplementSystem sys;
  WorkspaceStats stats;
  std::vector<Workspace> workspaces;
  std::unique_ptr<Solver> solver;
  std::vector<KKT_SystemAssembler> kkt;
  Eigen::VectorXd memory;
  bool is_initialized = false;
};

DenseMatrix GetFeasibleObjective(Program* prog);
bool Solve(const DenseMatrix& b, Program& prog,
           const SolverConfiguration& config, double* primal_variable);

}  // namespace conex
