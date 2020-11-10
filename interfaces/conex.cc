#include "conex.h"

#include <iostream>
#include <vector>
#include <memory>

#include <Eigen/Dense>

#include "conex/linear_constraint.h"
#include "conex/cone_program.h"
#include "conex/dense_lmi_constraint.h"
#include "conex/hermitian_psd.h"
#include "conex/constraint.h"

// TODO(FrankPermenter): check for null pointers.

#define CONEX_DEMAND(x, msg) \
    if (!(x)) { \
      std::cout << "Conex error: " << msg << std::endl; \
      return CONEX_FAILURE; \
    } 

#define SAFER_CAST_TO_Program(x, prog) \
    CONEX_DEMAND(x, "Program pointer is null.");\
    prog = static_cast<Program*>(x); \
    if (prog->is_initialized) { \
      if (prog->constraints.size() + 2 != prog->workspaces.size()) { \
        CONEX_DEMAND(false, "Program corrupted or invalid pointer."); \
      } \
    } else { \
      if (prog->workspaces.size() != 0) { \
        CONEX_DEMAND(false, "Program corrupted or invalid pointer."); \
      } \
    } \
    CONEX_DEMAND(prog, "Program corrupted or invalid pointer."); 



using DenseMatrix = Eigen::MatrixXd;
int ConexSolve(void* prog_ptr, const double*b, int br, const ConexSolverConfiguration*
               config, double* y, int yr) {
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

void ConexGetDualVariable(void* prog_ptr, int i, double* x, int xr,  int xc) {
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

int CONEX_UpdateLinearOperator(void* p, int constraint, int variable, int
                                hyper_complex_dim, int value, int row, int col) {
//  CONEX_DEMAND(p);
//  CONEX_DEMAND(hyper_complex_dim == 1 || hyper_complex_dim == 2 || 
//               hyper_complex_dim == 4 || hyper_complex_dim == 8);
//
  return CONEX_SUCCESS;
}

class Test {
 public:
  virtual void blah()  { std::cout << m_; };
  int m_ = 3;
};

CONEX_STATUS CONEX_AddLinearMatrixInequality(void* p, int order, int hyper_complex_dim, int* constraint_id) {
  CONEX_DEMAND(order >= 1, "Invalid LMI dimensions.");
  CONEX_DEMAND(constraint_id, "Received output null pointer.");
  CONEX_DEMAND(hyper_complex_dim == 1 || hyper_complex_dim == 2 || 
               hyper_complex_dim == 4 || hyper_complex_dim == 8, "Hypercomplex dimension must be 1, 2, 4, or 8.");

  Program* prg;
  SAFER_CAST_TO_Program(p, prg);

  switch (hyper_complex_dim) {
    case 1:
      prg->constraints.push_back(HermitianPsdConstraint<Real>(order));
      break;
    case 2:
      prg->constraints.push_back(HermitianPsdConstraint<Complex>(order));
      break;
    case 4:
      prg->constraints.push_back(HermitianPsdConstraint<Quaternions>(order));
      break;
    case 8:
      CONEX_DEMAND(order <= 3, "Order of octonion algebra cannot be greater than 3.");
      prg->constraints.push_back(HermitianPsdConstraint<Octonions>(order));
  }
  *constraint_id = prg->NumberOfConstraints() - 1;
  return CONEX_SUCCESS;
}
