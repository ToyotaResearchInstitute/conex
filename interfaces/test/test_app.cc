#include <iostream>
#include "../conex.h"

void SolveLP() {
  void* p = CONEX_CreateConeProgram();

  int status;
  int constraint_id = 0;
  int hyper_complex_dim = 1;
  int num_vars = 10;
  int order = num_vars;

  status = CONEX_SetNumberOfVariables(p, num_vars);

  status = CONEX_NewLinearMatrixInequality(p, order, hyper_complex_dim,
                                           &constraint_id);
  double b[num_vars];
  double y[num_vars];

  for (int i = 0; i < num_vars; i++) {
    status = CONEX_UpdateLinearOperator(p, constraint_id, .3, i, i, i, 0);
    b[i] = 1;
  }

  status = CONEX_UpdateAffineTerm(p, constraint_id, .3, 0, 0, 0);

  std::cout << status;
  CONEX_SolverConfiguration config;
  CONEX_SetDefaultOptions(&config);
  CONEX_Maximize(p, &b[0], num_vars, &config, &y[0], num_vars);

  CONEX_DeleteConeProgram(p);
}

void SolveQP() {
  void* p = CONEX_CreateConeProgram();

  int status;
  int constraint_id = 0;
  int num_vars = 4;

  double y[num_vars];

  status = CONEX_SetNumberOfVariables(p, num_vars);
  status = CONEX_NewQuadraticCost(p, &constraint_id);

  for (int i = 0; i < num_vars; i++) {
    status = CONEX_UpdateQuadraticCostMatrix(p, constraint_id, 1, i, i);
  }

  status = CONEX_NewLinearInequality(p, 1, &constraint_id);
  int row = 0;
  for (int i = 0; i < num_vars; i++) {
    int var = i;
    status = CONEX_UpdateLinearOperator(p, constraint_id, 1, var, row, 0, 0);
  }
  status = CONEX_UpdateAffineTerm(p, constraint_id, 1, row, 0, 0);

  std::cout << status;
  CONEX_SolverConfiguration config;
  CONEX_SetDefaultOptions(&config);
  config.enable_rescaling = 0;
  config.enable_line_search = 1;
  CONEX_Solve(p, &config, &y[0], num_vars);
  CONEX_DeleteConeProgram(p);
}

int main() {
  SolveLP();
  SolveQP();
  return 0;
}
