#ifndef CONEX_API_H
#define CONEX_API_H
#ifdef __cplusplus
extern "C" {
#endif

typedef int CONEX_STATUS;
enum { CONEX_SUCCESS = 0, CONEX_FAILURE = 1 };

typedef struct {
  int prepare_dual_variables;
  int initialization_mode;
  double inv_sqrt_mu_max;
  double minimum_mu;
  double maximum_mu;
  double divergence_upper_bound;
  int enable_line_search;
  double dinf_upper_bound;
  int final_centering_steps;
  double final_centering_tolerance;
  int initial_centering_steps_warmstart;
  int initial_centering_steps_coldstart;
  double warmstart_abort_threshold;
  int max_iterations;
  int iterative_refinement_iterations;
  double infeasibility_threshold;
  double kkt_error_tolerance;
  int enable_rescaling;
  int kkt_solver;
} CONEX_SolverConfiguration;

typedef struct {
  double mu;
  int iteration_number;
} CONEX_IterationStats;

typedef struct {
  int iterations;
} CONEX_SolutionStats;

void* CONEX_CreateConeProgram();
void CONEX_DeleteConeProgram(void*);

int CONEX_AddDenseLinearConstraint(void* prog, const double* A, int Ar, int Ac,
                                   const double* c, int cr);

int CONEX_AddLinearInequalities(void* prog, const double* A, int Ar, int Ac,
                                const double* lb, int num_lb, const double* ub,
                                int num_ub);

int CONEX_AddQuadraticCost(void* prog, const double* A, int Ar, int Ac);
//  Parameters Aarrayr, Aarrayc, cr, cc all equal the
//  order n of LMI.
// TODO(FrankPermenter): update this.
int CONEX_AddDenseLMIConstraint(void* prog, const double* Aarray, int Aarrayr,
                                int Aarrayc, int m, const double* cmat, int cr,
                                int cc);

int CONEX_AddSparseLMIConstraint(void* prog, const double* Aarray, int Aarrayr,
                                 int Aarrayc, int m, const double* cmat, int cr,
                                 int cc, const long* vars, int vars_c);

int CONEX_Maximize(void* prog, const double* b, int br,
                   const CONEX_SolverConfiguration* config, double* y, int yr);

int CONEX_Solve(void* prog, const CONEX_SolverConfiguration* config, double* y,
                int yr);

void CONEX_GetDualVariable(void* prog, int i, double* x, int xr, int xc);

int CONEX_GetDualVariableSize(void* prog_ptr, int i);

void CONEX_SetDefaultOptions(CONEX_SolverConfiguration* config);

void CONEX_GetIterationStats(void* prog, CONEX_IterationStats* stats,
                             int iter_num);

CONEX_STATUS CONEX_UpdateLinearOperator(void* program, int constraint,
                                        double value, int variable, int row,
                                        int col, int hyper_complex_dim);

CONEX_STATUS CONEX_NewLinearMatrixInequality(void* program, int order,
                                             int hyper_complex_dim,
                                             int* constraint_id);

CONEX_STATUS CONEX_UpdateAffineTerm(void* program, int constraint, double value,
                                    int row, int col, int hyper_complex_dim);

CONEX_STATUS CONEX_NewLorentzConeConstraint(void* program, int order,
                                            int* constraint_id);

CONEX_STATUS CONEX_NewLinearInequality(void* program, int num_rows,
                                       int* constraint_id);

CONEX_STATUS CONEX_NewQuadraticCost(void* p, int* constraint_id);
CONEX_STATUS CONEX_UpdateQuadraticCostMatrix(void* p, int id, double value,
                                             int row, int col);

CONEX_STATUS CONEX_SetNumberOfVariables(void* program, int m);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
