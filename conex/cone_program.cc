#include "conex/cone_program.h"

#include <vector>

#include "conex/divergence.h"
#include "conex/newton_step.h"

namespace conex {


double CalcMinMu(double lambda_max, double, MuSelectionParameters* p) {
  const double kMaxNormInfD = p->limit;
  double inv_sqrt_mu = (1.0 + kMaxNormInfD) / (lambda_max + 1e-12);
  if (inv_sqrt_mu < 1e-3) {
    inv_sqrt_mu = 1e-3;
  }
  return inv_sqrt_mu;
}

template <typename T>
void SetIdentity(std::vector<T>* c) {
  for (auto& ci : *c) {
    SetIdentity(&ci);
  }
}

template <typename T>
void TakeStep(std::vector<T>* c, const StepOptions& newton_step_parameters,
              const Ref& y, StepInfo* info) {
  StepInfo info_i;
  info->normsqrd = 0;
  info->norminfd = -1;
  for (auto& ci : *c) {
    TakeStep(&ci, newton_step_parameters, y, &info_i);
    if (info_i.norminfd > info->norminfd) {
      info->norminfd = info_i.norminfd;
    }
    info->normsqrd += info_i.normsqrd;
  }
}

template <typename T>
void GetMuSelectionParameters(std::vector<T>* c, const Ref& y,
                              MuSelectionParameters* p) {
  p->gw_norm_squared = 0;
  p->gw_lambda_max = -1000;
  for (auto& ci : *c) {
    GetMuSelectionParameters(&ci, y, p);
  }
}

template <typename T>
int Rank(const std::vector<T>& c) {
  int rank = 0;
  for (const auto& ci : c) {
    rank += Rank(ci);
  }
  return rank;
}

template <typename T>
void ConstructSchurComplementSystem(std::vector<T>* c, bool initialize,
                                    SchurComplementSystem* sys) {
  bool init = initialize;
  for (auto& ci : *c) {
    ConstructSchurComplementSystem(&ci, init, sys);
    init = false;
  }
}

void Initialize(Program& prog, const SolverConfiguration& config) {
  if (config.initialization_mode == 0) {
    prog.InitializeWorkspace();
  } else {
    if (!prog.is_initialized) {
      std::cerr << "Cannot warmstart without coldstart.";
      assert(0);
    }
  }

  if (config.initialization_mode == 0) {
    SetIdentity(&prog.constraints);
  }
  return;
}
double UpdateMu(std::vector<Constraint>& constraints,
                const Eigen::LLT<Eigen::Ref<DenseMatrix>>& llt,
                const SchurComplementSystem& sys, const DenseMatrix& b,
                const SolverConfiguration& config, int rankK,
                Ref* temporary_memory) {
  MuSelectionParameters mu_param;
  (*temporary_memory) = sys.AQc - b;
  llt.solveInPlace(*temporary_memory);
  GetMuSelectionParameters(&constraints, *temporary_memory, &mu_param);

  return DivergenceUpperBoundInverse(
      config.divergence_upper_bound * rankK, mu_param.gw_norm_squared,
      mu_param.gw_lambda_max, mu_param.gw_trace, rankK);
}

void ApplyLimits(double* x, double lb, double ub) {
  if (*x > ub) {
    *x = ub;
  }

  if (*x < lb) {
    *x = lb;
  }
}
bool Solve(const DenseMatrix& b, Program& prog,
           const SolverConfiguration& config, double* primal_variable) {
#ifdef EIGEN_USE_MKL_ALL
  std::cout << "CONEX: MKL Enabled";
#endif

  auto& constraints = prog.constraints;
  auto& sys = prog.sys;

  bool solved = 1;

  std::cout.precision(2);
  std::cout << std::scientific;

  int m = b.rows();
  prog.SetNumberOfVariables(m);

  // Empty program
  if (prog.constraints.size() == 0) {
    Eigen::Map<DenseMatrix> ynan(primal_variable, m, 1);
    solved = 0;
    ynan.array() = b.array() * std::numeric_limits<double>::infinity();
    return solved;
  }

  Initialize(prog, config);

  Eigen::MatrixXd ydata(m, 1);
  Eigen::Map<DenseMatrix> yout(primal_variable, m, 1);
  Ref y(ydata.data(), m, 1);

  double inv_sqrt_mu_max = config.inv_sqrt_mu_max;

  StepOptions newton_step_parameters;
  newton_step_parameters.affine = 0;
  IterationStats stats;
  newton_step_parameters.inv_sqrt_mu = 0;
  newton_step_parameters.affine = false;

  int rankK = Rank(constraints);
  int centering_steps = 0;

  for (int i = 0; i < config.max_iterations; i++) {
    bool update_mu =
        (i == 0) || ((newton_step_parameters.inv_sqrt_mu < inv_sqrt_mu_max) &&
                     i < config.max_iterations - config.final_centering_steps);

    if (!update_mu && (centering_steps >= config.final_centering_steps)) {
      break;
    }

    ConstructSchurComplementSystem(&constraints, true /*init*/, &sys);

    Eigen::LLT<Eigen::Ref<DenseMatrix>> llt(sys.G);
    if (llt.info() != Eigen::Success) {
      PRINTSTATUS("LLT FAILURE.");
      return false;
      break;
    }

    if (update_mu) {
      START_TIMER(Mu)
      newton_step_parameters.inv_sqrt_mu =
          UpdateMu(constraints, llt, sys, b, config, rankK, &y);
      END_TIMER
    } else {
      centering_steps++;
    }

    const double max = config.inv_sqrt_mu_max;
    const double min = std::sqrt(1.0 / (1e-15 + config.maximum_mu));
    ApplyLimits(&newton_step_parameters.inv_sqrt_mu, min, max);

    double mu = 1.0 / (newton_step_parameters.inv_sqrt_mu);
    mu *= mu;

    if (mu > config.infeasibility_threshold) {
      if (i > 3) {
        solved = 0;
        PRINTSTATUS("Infeasible Or Unbounded.");
        return solved;
      }
    }

    y = newton_step_parameters.inv_sqrt_mu * (b + sys.AQc) - 2 * sys.AW;
    llt.solveInPlace(y);
    newton_step_parameters.e_weight = 1;
    newton_step_parameters.c_weight = newton_step_parameters.inv_sqrt_mu;

    StepInfo info;
    START_TIMER(Step)
    TakeStep(&constraints, newton_step_parameters, y, &info);
    END_TIMER

    const double d_2 = std::sqrt(std::fabs(info.normsqrd));
    const double d_inf = std::fabs(info.norminfd);

    if (i < 10) {
      std::cout << "i:  " << i << ", ";
    } else {
      std::cout << "i: " << i << ", ";
    }
    REPORT(mu);
    REPORT(d_2);
    REPORT(d_inf);

    prog.stats.num_iter = i + 1;
    prog.stats.sqrt_inv_mu[i] = newton_step_parameters.inv_sqrt_mu;
    std::cout << std::endl;
  }

  if (config.prepare_dual_variables) {
    DenseMatrix y2;
    bool iter_refine = false;
    if (iter_refine) {
      ConstructSchurComplementSystem(&constraints, true, &sys);
      Eigen::LLT<Eigen::Ref<DenseMatrix>> llt(sys.G);
      DenseMatrix L = llt.matrixL();
      DenseMatrix bres = newton_step_parameters.inv_sqrt_mu * b - 1 * sys.AW;
      y2 = bres * 0;
      for (int i = 0; i < 1; i++) {
        y2 += llt.solve(bres - L * L.transpose() * y2);
      }
    } else {
      ConstructSchurComplementSystem(&constraints, true, &sys);
      Eigen::LLT<Eigen::Ref<DenseMatrix>> llt(sys.G);
      DenseMatrix bres = newton_step_parameters.inv_sqrt_mu * b - 1 * sys.AW;
      y2 = llt.solve(bres);
    }

    newton_step_parameters.affine = true;
    newton_step_parameters.e_weight = 0;
    newton_step_parameters.c_weight = 0;
    Ref y2map(y2.data(), y2.rows(), y2.cols());
    StepInfo info;
    TakeStep(&constraints, newton_step_parameters, y2map, &info);
  }
  y /= newton_step_parameters.inv_sqrt_mu;
  yout = y;
  solved = 1;

  PRINTSTATUS("Solved.");
  return solved;
}

DenseMatrix GetFeasibleObjective(int m, std::vector<Constraint>& constraints) {
  std::vector<Workspace> workspaces;
  for (auto& constraint : constraints) {
    workspaces.push_back(constraint.workspace());
  }

  WorkspaceSchurComplement sys{m};
  workspaces.push_back(Workspace{&sys});
  int size_constraints = SizeOf(workspaces);

  Eigen::VectorXd memory(size_constraints);
  Initialize(&workspaces, &memory[0]);

  SetIdentity(&constraints);
  ConstructSchurComplementSystem(&constraints, true, &sys);
  return .5 * sys.AW;
}

} // namespace conex
