#include "conex.h"
#include <iostream>


int main() {
  void* p = ConexCreateConeProgram();

  int status;
  int constraint_id = 0;
  int hyper_complex_dim = 1;
  int num_vars = 200;
  int order = num_vars;

  status = CONEX_NewLinearMatrixInequality(p, order, hyper_complex_dim, 
                                              &constraint_id);
  double b[num_vars];
  double y[num_vars];
 
  for (int i = 0; i < num_vars; i++) {
    status = CONEX_UpdateLinearOperator(p, constraint_id, .3, i, i, i, 0);
    b[i] = 1;
  }

  status = CONEX_UpdateAffineTerm(p, constraint_id, .3,  0, 0, 0);

  std::cout << status;
  ConexSolverConfiguration config;
  ConexSetDefaultOptions(&config);
  ConexSolve(p, &b[0], num_vars, &config, &y[0], num_vars);

  ConexDeleteConeProgram(p);
}
