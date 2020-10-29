#ifndef CONEX_API_H
#define CONEX_API_H
#ifdef __cplusplus
extern "C"
{
#endif
typedef double Real;
typedef struct {
  int prepare_dual_variables;
  int max_iterations;
  double inv_sqrt_mu_max;
  double divergence_upper_bound;
  int final_centering_steps;
  double infeasibility_threshold; 
  int collect_statistics;
} ConexSolverConfiguration;

typedef struct {
  double mu;
  int iteration_number;
} ConexIterationStats;

typedef struct {
  int iterations;
} ConexSolutionStats;


void* ConexCreateConeProgram();
void ConexDeleteConeProgram(void*);

int ConexAddDenseLinearConstraint(void* prog,
  const double* A, int Ar, int Ac,
  const double* c, int cr);


//  Parameters Aarrayr, Aarrayc, cr, cc all equal the
//  order n of LMI.
// TODO(FrankPermenter): update this.
int ConexAddDenseLMIConstraint(void* prog,
  const double* Aarray, int Aarrayr, int Aarrayc, int m,
  const double* cmat, int cr, int cc);

int ConexAddSparseLMIConstraint(void* prog,
  const double* Aarray, int Aarrayr, int Aarrayc, int m,
  const double* cmat, int cr, int cc,
  const long* vars, int vars_c);

int ConexSolve(void* prog, const double*b, int br, const ConexSolverConfiguration* config, 
                double* y, int yr);

void ConexGetDualVariable(void* prog, int i, double* x, int xr, int xc);

int ConexGetDualVariableSize(void* prog_ptr, int i);

void ConexSetDefaultOptions(ConexSolverConfiguration* config);

void ConexGetIterationStats(void* prog, ConexIterationStats* stats, int iter_num);

#ifdef __cplusplus
} // extern "C"
#endif
#endif
