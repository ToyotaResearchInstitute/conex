#include <iostream>
#include "../conex.h"

int main() {
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
  CONEX_Solve(p, &b[0], num_vars, &config, &y[0], num_vars);

  CONEX_DeleteConeProgram(p);
}
