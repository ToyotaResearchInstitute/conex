#include "swig/conex.h"

#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include "conex/linear_constraint.h"
#include "conex/cone_program.h"
#include "conex/dense_lmi_constraint.h"
#include "conex/constraint.h"

using DenseMatrix = Eigen::MatrixXd;
using ConexConeProgram = void*;

int ConexSolve(void* prog_ptr, const Real*b, int br, const ConexSolverConfiguration* config,
           Real* y, int yr) {
  using InputMatrix = Eigen::Map<const DenseMatrix>;
  InputMatrix bmap(b, br, 1);
  DenseMatrix blinear = bmap;

  Program& prog = *reinterpret_cast<Program*>(prog_ptr);

  SolverConfiguration c;
  c.prepare_dual_variables = config->prepare_dual_variables;
  c.max_iter = config->max_iter;
  c.inv_sqrt_mu_max = config->inv_sqrt_mu_max;
  c.dinf_limit = config->dinf_limit;
  c.final_centering_steps = config->final_centering_steps;
  c.convergence_rate_threshold = config->convergence_rate_threshold;
  c.divergence_threshold = config->divergence_threshold;

  return Solve(blinear, prog, c, y);
}

void ConexGetDualVariable(void* prog_ptr, int i, Real* x, int xr,  int xc) {
  Program& prog = *reinterpret_cast<Program*>(prog_ptr);
  assert(prog.constraints.at(i).dual_variable_size() == xr * xc);

  using InputMatrix = Eigen::Map<DenseMatrix>;
  double sqrt_inv_mu = prog.stats.sqrt_inv_mu[prog.stats.num_iter - 1];

  prog.constraints.at(i).get_dual_variable(x);

  InputMatrix xmap(x, xr, xc);
  xmap.array() /= prog.stats.sqrt_inv_mu[prog.stats.num_iter - 1];
}

int ConexGetDualVariableSize(void* prog_ptr, int i) {
  using InputMatrix = Eigen::Map<DenseMatrix>;
  Program& prog = *reinterpret_cast<Program*>(prog_ptr);
  return prog.constraints.at(i).dual_variable_size();
}

void* ConexCreateConeProgram() {
  return reinterpret_cast<void*>(new Program);
}

void ConexDeleteConeProgram(ConexConeProgram prog) {
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
  std::vector<InputMatrix> Avect;
  for (int i = 0; i < m; i++) {
    InputMatrix Amap(offset, Ar, Ac);
    Avect.push_back(Amap);
    offset += Ar * Ac;
  }
  InputMatrix Cmap(c, cr, cc);

  int n = cc;

  DenseLMIConstraint T3{n,  &Avect, &Cmap};
  auto& program = *reinterpret_cast<Program*>(prog);
  int constraint_id = program.constraints.size();
  program.constraints.push_back(T3);
  return constraint_id;
}

int ConexAddDenseLinearConstraint(ConexConeProgram fred,
  const double* A, int Ar, int Ac,
  const double* c, int cr) {
  assert(Ar == cr);

  int n = Ar;
  int m = Ac;

  LinearConstraint T3{n, m, A, c};
  auto& program = *reinterpret_cast<Program*>(fred);

  int constraint_id = program.constraints.size();
  program.constraints.push_back(T3);
  return constraint_id;
}


ConexSolverConfiguration ConexDefaultOptions() {
  ConexSolverConfiguration c;
  SolverConfiguration config;
  c.prepare_dual_variables = config.prepare_dual_variables;
  c.max_iter = config.max_iter;
  c.inv_sqrt_mu_max = config.inv_sqrt_mu_max;
  c.dinf_limit = config.dinf_limit;
  c.final_centering_steps = config.final_centering_steps;
  c.convergence_rate_threshold = config.convergence_rate_threshold;
  c.divergence_threshold = config.divergence_threshold;
  return c;
}



