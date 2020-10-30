#include "conex.h"

#include <iostream>
#include <vector>
#include <memory>

#include <Eigen/Dense>

#include "conex/linear_constraint.h"
#include "conex/cone_program.h"
#include "conex/dense_lmi_constraint.h"
#include "conex/constraint.h"

// TODO(FrankPermenter): check for null pointers.

using DenseMatrix = Eigen::MatrixXd;
int ConexSolve(void* prog_ptr, const Real*b, int br, const ConexSolverConfiguration*
               config, Real* y, int yr) {
  using InputMatrix = Eigen::Map<const DenseMatrix>;
  InputMatrix bmap(b, br, 1);
  DenseMatrix blinear = bmap;

  SolverConfiguration c;
  c.prepare_dual_variables = config->prepare_dual_variables;
  c.max_iterations = config->max_iterations;
  c.inv_sqrt_mu_max = config->inv_sqrt_mu_max;
  c.divergence_upper_bound = config->divergence_upper_bound;
  c.final_centering_steps = config->final_centering_steps;
  c.infeasibility_threshold = config->infeasibility_threshold;
  c.minimum_mu = config->minimum_mu;
  c.maximum_mu = config->maximum_mu;
  c.initialization_mode = config->initialization_mode;

  Program& prog = *reinterpret_cast<Program*>(prog_ptr);

  return Solve(blinear, prog, c, y);
}

void ConexGetDualVariable(void* prog_ptr, int i, Real* x, int xr,  int xc) {
  Program& prog = *reinterpret_cast<Program*>(prog_ptr);
  assert(prog.constraints.at(i).dual_variable_size() == xr * xc);

  using InputMatrix = Eigen::Map<DenseMatrix>;

  prog.constraints.at(i).get_dual_variable(x);

  InputMatrix xmap(x, xr, xc);
  xmap.array() /= prog.stats.sqrt_inv_mu[prog.stats.num_iter - 1];
}

int ConexGetDualVariableSize(void* prog_ptr, int i) {
  Program& prog = *reinterpret_cast<Program*>(prog_ptr);
  return prog.constraints.at(i).dual_variable_size();
}

void* ConexCreateConeProgram() {
  return reinterpret_cast<void*>(new Program);
}

void ConexDeleteConeProgram(void* prog) {
  delete reinterpret_cast<Program*>(prog);
}

int ConexAddDenseLMIConstraint(void* prog,
  const double* A, int Ar, int Ac, int m,
  const double* c, int cr, int cc) {
  assert(Ar == Ac);
  assert(Ar == cr);
  assert(cc == cr);

  using InputMatrix = Eigen::Map<const DenseMatrix>;
  auto offset = A;
  std::vector<DenseMatrix> Avect;
  for (int i = 0; i < m; i++) {
    InputMatrix Amap(offset, Ar, Ac);
    Avect.push_back(Amap);
    offset += Ar * Ac;
  }
  InputMatrix Cmap(c, cr, cc);

  int n = cc;

  DenseLMIConstraint T3{n,  Avect, Cmap};
  auto& program = *reinterpret_cast<Program*>(prog);
  int constraint_id = program.constraints.size();
  program.constraints.push_back(T3);
  return constraint_id;
}

int ConexAddSparseLMIConstraint(void* prog,
  const double* A, int Ar, int Ac, int num_vars,
  const double* c, int cr, int cc,
  const long *vars, int vars_rows) {
  assert(Ar == Ac);
  assert(Ar == cr);
  assert(cc == cr);
  assert(vars_rows == num_vars);

  // TODO(FrankPermenter): Remove these copies.
  using InputMatrix = Eigen::Map<const DenseMatrix>;
  auto offset = A;
  std::vector<DenseMatrix> Avect;
  std::vector<int> variables(num_vars);
  for (int i = 0; i < num_vars; i++) {
    InputMatrix Amap(offset, Ar, Ac);
    Avect.push_back(Amap);
    offset += Ar * Ac;
    variables.at(i) = *(vars + i);
  }
  InputMatrix Cmap(c, cr, cc);


  SparseLMIConstraint T3{Avect, Cmap, variables};
  auto& program = *reinterpret_cast<Program*>(prog);
  int constraint_id = program.constraints.size();
  program.constraints.push_back(T3);
  return constraint_id;
}

int ConexAddDenseLinearConstraint(void* prog,
  const double* A, int Ar, int Ac,
  const double* c, int cr) {
  assert(Ar == cr);

  int n = Ar;
  int m = Ac;

  LinearConstraint T3{n, m, A, c};
  auto& program = *reinterpret_cast<Program*>(prog);

  int constraint_id = program.constraints.size();
  program.constraints.push_back(T3);
  return constraint_id;
}

void ConexSetDefaultOptions(ConexSolverConfiguration* c) {
  if (c == NULL) {
    std::cerr << "Received null pointer.";
    return;
  }
  SolverConfiguration config;
  c->prepare_dual_variables = config.prepare_dual_variables;
  c->max_iterations = config.max_iterations;
  c->inv_sqrt_mu_max = config.inv_sqrt_mu_max;
  c->maximum_mu = config.maximum_mu; 
  c->minimum_mu = config.minimum_mu;
  c->divergence_upper_bound = config.divergence_upper_bound;
  c->final_centering_steps = config.final_centering_steps;
  c->infeasibility_threshold = config.infeasibility_threshold;
  c->initialization_mode = config.initialization_mode;
}

void ConexGetIterationStats(void* prog, ConexIterationStats* stats, int iter_num_circular) {
  if ((prog == NULL) || (stats == NULL)) {
    std::cerr << "Received null pointer.";
    return;
  }

  auto& program = *reinterpret_cast<Program*>(prog);

  if (!program.stats.IsInitialized()) {
    std::cerr << "No statistics available.";
    return;
  }

  int iter_num = iter_num_circular;
  if (iter_num_circular < 0) {
    iter_num = program.stats.num_iter + iter_num_circular;
  }

  if ((program.stats.num_iter <= iter_num) || (iter_num < 0)) {
    std::cerr << "Specified iteration is out of bounds.";
    return;
  }
  stats->mu = 1.0/(program.stats.sqrt_inv_mu[iter_num] *
                            program.stats.sqrt_inv_mu[iter_num]); 
  stats->iteration_number = iter_num;
}
