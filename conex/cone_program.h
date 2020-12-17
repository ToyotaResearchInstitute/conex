#pragma once
#include "conex/constraint_manager.h"
#include "conex/kkt_assembler.h"
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
  void SetNumberOfVariables(int m) {
    kkt_system_manager_.SetNumberOfVariables(m);
    sys.m_ = m;
  }

  void InitializeWorkspace() {
    for (auto& constraint : kkt_system_manager_.eqs) {
      workspaces.push_back(constraint.constraint.workspace());
    }
    workspaces.push_back(Workspace{&stats});
    workspaces.push_back(Workspace{&sys});
    memory.resize(SizeOf(workspaces));
    Initialize(&workspaces, &memory[0]);

    is_initialized = true;
  }

  template <typename T>
  void AddConstraint(T&& d) {
    // constraints.push_back(d);
    // kkt_system_manager_.AddConstraint(d.kkt_assembler());
    kkt_system_manager_.AddConstraint(d);
  }

  template <typename T>
  void AddConstraint(T&& d, const std::vector<int>& variables) {
    kkt_system_manager_.AddConstraint(d, variables);
    // constraints.push_back(d);
  }

  // std::vector<Constraint> constraints;

  int NumberOfConstraints() { return kkt_system_manager_.eqs.size(); }

  ConstraintManager<Container> kkt_system_manager_;
  std::vector<Constraint*> constraints;
  SchurComplementSystem sys;
  WorkspaceStats stats;
  std::vector<Workspace> workspaces;
  Eigen::VectorXd memory;
  bool is_initialized = false;
};

DenseMatrix GetFeasibleObjective(int m, std::vector<Constraint*>& constraints);
bool Solve(const DenseMatrix& b, Program& prog,
           const SolverConfiguration& config, double* primal_variable);

}  // namespace conex
