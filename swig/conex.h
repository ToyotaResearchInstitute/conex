#ifndef CONEX_API_H
#define CONEX_API_H
#ifdef __cplusplus
extern "C"
{
#endif
typedef double Real;
typedef struct {
  int prepare_dual_variables;
  int max_iter;
  double inv_sqrt_mu_max;
  double dinf_limit;
  int final_centering_steps;
  double convergence_rate_threshold;
  double divergence_threshold; 
} ConexSolverConfiguration;



void* ConexCreateConeProgram();
void ConexDeleteConeProgram(void*);

int ConexAddDenseLinearConstraint(void* prog, 
  const double* A, int Ar, int Ac,
  const double* c, int cr);

int ConexAddDenseLMIConstraint(void* prog, 
  const double* Aarray, int Aarrayr, int Aarrayc, int m,
  const double* cmat, int cr, int cc);

int ConexSolve(void* prog, const double*b, int br, const ConexSolverConfiguration* config, 
                double* y, int yr);

void ConexGetDualVariable(void* prog, int i, double* x, int xr, int xc);

ConexSolverConfiguration ConexDefaultOptions() {
  ConexSolverConfiguration config;
  config.prepare_dual_variables = 0;
  config.max_iter = 100;
  config.inv_sqrt_mu_max = 1000;
  config.dinf_limit = .95;
  config.final_centering_steps = 5;
  config.convergence_rate_threshold = .5;
  config.divergence_threshold = 100;
  return config;
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif
