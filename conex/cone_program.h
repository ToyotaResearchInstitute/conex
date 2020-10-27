#pragma once
#include "constraint.h"
#include "workspace.h"

//  Allocates memory for internal computations.

struct SolverConfiguration {
  int prepare_dual_variables = 0;
  double inv_sqrt_mu_max = 1000;
  double divergence_upper_bound = 1;
  int final_centering_steps = 5;
  int max_iterations = 25;
  double infeasibility_threshold = 1e5; 
};

class Program {
 public:
  void SetNumberOfVariables(int m) {
    sys.m_ = m;
  }
  void InitializeWorkspace() {
    for (auto& constraint : constraints) {
      workspaces.push_back(constraint.workspace());
    }
    workspaces.push_back(Workspace{&stats});
    workspaces.push_back(Workspace{&sys});
    memory.resize(SizeOf(workspaces));
    Initialize(&workspaces, &memory[0]);

  }

  std::vector<Constraint> constraints;

  SchurComplementSystem sys;
  WorkspaceStats stats;
  std::vector<Workspace> workspaces;
  Eigen::VectorXd memory; 
};

class ConvexProgram {
  void AddQuadraticCost() {
    num_epigraph++;
  }

  void AddLinearCost() {

  }

 private:
  Program prog;
  int num_epigraph;
};

class PolynomialProgram {
  void SetBasis(int m);
  void IsSumOfSquares(int c);

  // f = m^T Q m
  // (a, a, b)
  // a x^2 + a x + b
  // Fa + b = A(Q)
  // F^T y = 0
};

DenseMatrix GetFeasibleObjective(int m, std::vector<Constraint>& constraints);
bool Solve(const DenseMatrix& b, Program& prog,  
           const SolverConfiguration& config,
           double* primal_variable);

