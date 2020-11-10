#ifndef CONEX_API_H
#define CONEX_API_H
#ifdef __cplusplus
extern "C"
{
#endif

typedef int CONEX_STATUS;
enum { CONEX_SUCCESS = 0, CONEX_FAILURE = 1};

typedef struct {
  int prepare_dual_variables;
  int max_iterations;
  double inv_sqrt_mu_max;
  double minimum_mu;
  double maximum_mu;
  double divergence_upper_bound;
  int final_centering_steps;
  double infeasibility_threshold; 
  double initialization_mode;
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

int ConexAddComplexLinearMatrixInequality(void* prog,
  const double* Aarray, int Aarrayr, int Aarrayc, int m,
  const double* cmat, int cr, int cc,
  const long* vars, int vars_c);

int ConexAddQuaternionLinearMatrixInequality(void* prog,
  const double* Aarray, int Aarrayr, int Aarrayc, int m,
  const double* cmat, int cr, int cc,
  const long* vars, int vars_c);

int ConexAddOctonionLinearMatrixInequality(void* prog,
  const double* A,   int r,  int c,  int m,
  const double* Ai, int ri, int ci, int mi,
  const double* Aj, int rj, int cj, int mj,
  const double* Ak, int rk, int ck, int mk,
  const double* Al, int rl, int cl, int ml,
  const double* Am, int rm, int cm, int mm,
  const double* An, int rn, int cn, int mn,
  const double* Ao, int ro, int co, int mo,
  const double* cmat, int cr, int cc,
  const long* vars, int vars_c);

int ConexSolve(void* prog, const double*b, int br, const ConexSolverConfiguration* config, 
                double* y, int yr);

void ConexGetDualVariable(void* prog, int i, double* x, int xr, int xc);

int ConexGetDualVariableSize(void* prog_ptr, int i);

// TODO(FrankPermenter): Rename Options to Configuration
void ConexSetDefaultOptions(ConexSolverConfiguration* config);

void ConexGetIterationStats(void* prog, ConexIterationStats* stats, int iter_num);



int NewSparseLinearMatrixInequality(void* prog, int order,
                                    int hyper_complex_dim);

int SetAffine(int constraint, int variable, int hyper_complex_dim,
              int value, int row, int col);

CONEX_STATUS CONEX_UpdateLinearOperator(int constraint, int variable, int hyper_complex_dim,
                               int value, int row, int col);

int CONEX_AddLinearMatrixInequality(void* program, int order, int  hyper_complex_dim, 
                                    int *constraint_id);


#ifdef __cplusplus
} // extern "C"
#endif
#endif
