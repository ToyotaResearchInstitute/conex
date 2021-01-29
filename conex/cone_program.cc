#include "conex/cone_program.h"
#include "conex/kkt_solver.h"

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
void SetIdentity(std::vector<T*>* c) {
  for (auto& ci : *c) {
    SetIdentity(ci);
  }
}

Eigen::VectorXd Vars(const Eigen::VectorXd& x, std::vector<int> indices) {
  Eigen::VectorXd z(indices.size());
  int cnt = 0;
  for (auto i : indices) {
    z(cnt++) = x(i);
  }
  return z;
}

void PrepareStep(ConstraintManager<Container>* kkt,
                 const StepOptions& newton_step_parameters, const Ref& y,
                 StepInfo* info) {
  StepInfo info_i;
  info_i.normsqrd = 0;
  info_i.norminfd = 0;
  info->normsqrd = 0;
  info->norminfd = -1;
  int i = 0;
  for (auto& ci : kkt->eqs) {
    // TODO(FrankPermenter): Remove creation of these maps.
    auto ysegment = Vars(y, kkt->cliques.at(i));
    Eigen::Map<Eigen::MatrixXd, Eigen::Aligned> z(ysegment.data(),
                                                  ysegment.size(), 1);
    PrepareStep(&ci.constraint, newton_step_parameters, z, &info_i);
    if (info_i.norminfd > info->norminfd) {
      info->norminfd = info_i.norminfd;
    }
    info->normsqrd += info_i.normsqrd;
    i++;
  }
}

void TakeStep(ConstraintManager<Container>* kkt,
              const StepOptions& newton_step_parameters) {
  for (auto& ci : kkt->eqs) {
    TakeStep(&ci.constraint, newton_step_parameters);
  }
}

void GetMuSelectionParameters(ConstraintManager<Container>* constraints,
                              const Ref& y, MuSelectionParameters* p) {
  p->gw_norm_squared = 0;
  p->gw_lambda_max = -1000;
  int i = 0;
  for (auto& ci : constraints->eqs) {
    auto ysegment = Vars(y, constraints->cliques.at(i));
    Eigen::Map<Eigen::MatrixXd, Eigen::Aligned> z(ysegment.data(),
                                                  ysegment.size(), 1);
    GetMuSelectionParameters(&ci.constraint, z, p);
    i++;
  }
}

template <typename T>
int Rank(const std::vector<T*>& c) {
  int rank = 0;
  for (const auto& ci : c) {
    rank += Rank(*ci);
  }
  return rank;
}

template <typename T>
void ConstructSchurComplementSystem(std::vector<T*>* c, bool initialize,
                                    SchurComplementSystem* sys) {
  bool init = initialize;
  for (auto& ci : *c) {
    ConstructSchurComplementSystem(ci, init, sys);
    init = false;
  }
}

bool Initialize(Program& prog, const SolverConfiguration& config) {
  if (!prog.is_initialized || config.initialization_mode == 0) {
    auto& solver = prog.solver;
    auto& kkt = prog.kkt;
    prog.InitializeWorkspace();
    if (config.initialization_mode == 0) {
      SetIdentity(&prog.constraints);
    }

    START_TIMER(Sparsity);
    solver = std::make_unique<Solver>(prog.kkt_system_manager_.cliques,
                                      prog.kkt_system_manager_.dual_vars);

    kkt.clear();
    for (auto& c : prog.kkt_system_manager_.eqs) {
      c.kkt_assembler.Reset();
    }

    for (auto& c : prog.kkt_system_manager_.eqs) {
      c.kkt_assembler.workspace_ = &c.constraint;
      kkt.push_back(&c.kkt_assembler);
    }
    solver->Bind(&kkt);
    END_TIMER
  }
  return true;
}

double UpdateMu(ConstraintManager<Container>& constraints,
                std::unique_ptr<Solver>& solver, const DenseMatrix& AQc,
                const DenseMatrix& b, const SolverConfiguration& config,
                int rankK, Ref* temporary_memory) {
  MuSelectionParameters mu_param;
  *temporary_memory = solver->Solve(AQc - b);
  GetMuSelectionParameters(&constraints, *temporary_memory, &mu_param);
  mu_param.rank = rankK;

  double divergence_bound = config.divergence_upper_bound * rankK;

  double inv_sqrt_mu = 0;
  inv_sqrt_mu = DivergenceUpperBoundInverse(divergence_bound, mu_param);

  // If inverse evaluation has failed, choose mu that minimizes the norm of the
  // Newton step.
  if (inv_sqrt_mu < 0) {
    inv_sqrt_mu = mu_param.gw_trace / mu_param.gw_norm_squared;
  }

  return inv_sqrt_mu;
}

void ApplyLimits(double* x, double lb, double ub) {
  if (*x > ub) {
    *x = ub;
  }

  if (*x < lb) {
    *x = lb;
  }
}

bool Solve(const DenseMatrix& bin, Program& prog,
           const SolverConfiguration& config, double* primal_variable) {
#ifdef EIGEN_USE_MKL_ALL
  std::cout << "CONEX: MKL Enabled";
#endif

  auto& constraints = prog.constraints;
  auto& solver = prog.solver;
  bool solved = 1;

#if CONEX_VERBOSE
  std::cout.precision(2);
  std::cout << std::scientific;
  std::cout << "Starting the Conex optimizer: \n";
#endif

  CONEX_DEMAND(prog.GetNumberOfVariables() == bin.rows(),
               "Cost vector dimension does not equal number of variables");

  int m = bin.rows();
  // Empty program
  if (prog.NumberOfConstraints() == 0) {
    Eigen::Map<DenseMatrix> ynan(primal_variable, m, 1);
    solved = 0;
    ynan.array() = bin.array() * std::numeric_limits<double>::infinity();
    return solved;
  }

  START_TIMER(Assemble)
  Initialize(prog, config);
  END_TIMER
  std::cout << "\n";

  Eigen::MatrixXd ydata(prog.kkt_system_manager_.SizeOfKKTSystem(), 1);
  Eigen::Map<DenseMatrix> yout(primal_variable, m, 1);
  Ref y(ydata.data(), prog.kkt_system_manager_.SizeOfKKTSystem(), 1);

  double inv_sqrt_mu_max = config.inv_sqrt_mu_max;

  StepOptions newton_step_parameters;
  newton_step_parameters.affine = 0;
  IterationStats stats;
  newton_step_parameters.inv_sqrt_mu = 0;
  newton_step_parameters.affine = false;

  int rankK = Rank(constraints);
  int centering_steps = 0;

  Eigen::VectorXd AW(prog.kkt_system_manager_.SizeOfKKTSystem());
  Eigen::VectorXd AQc(prog.kkt_system_manager_.SizeOfKKTSystem());
  Eigen::VectorXd b(prog.kkt_system_manager_.SizeOfKKTSystem());
  b.setZero();
  b.head(m) << bin;

  for (int i = 0; i < config.max_iterations; i++) {
#if CONEX_VERBOSE
    if (i < 10) {
      std::cout << "i:  " << i << ", ";
    } else {
      std::cout << "i: " << i << ", ";
    }
#endif
    bool update_mu =
        (i == 0) || ((newton_step_parameters.inv_sqrt_mu < inv_sqrt_mu_max) &&
                     i < config.max_iterations - config.final_centering_steps);

    if (!update_mu && (centering_steps >= config.final_centering_steps)) {
      break;
    }

    START_TIMER(Assemble)
    solver->Assemble(&AW, &AQc);
    END_TIMER

    START_TIMER(Factor)
    if (!solver->Factor()) {
      solver->Assemble(&AW, &AQc);
      solver->Factor();
      solved = 0;
      PRINTSTATUS("Factorization failed.");
      return solved;
    }
    END_TIMER

    if (update_mu) {
      newton_step_parameters.inv_sqrt_mu =
          UpdateMu(prog.kkt_system_manager_, solver, AQc, b, config, rankK, &y);
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
    y = newton_step_parameters.inv_sqrt_mu * (b + AQc) - 2 * AW;
    START_TIMER(Solve)
    solver->SolveInPlace(&y);
    END_TIMER

    newton_step_parameters.e_weight = 1;
    newton_step_parameters.c_weight = newton_step_parameters.inv_sqrt_mu;

    StepInfo info;
    START_TIMER(Update)
    PrepareStep(&prog.kkt_system_manager_, newton_step_parameters, y, &info);
    newton_step_parameters.step_size = 2.0 / info.norminfd * info.norminfd;
    if (newton_step_parameters.step_size > 1) {
      newton_step_parameters.step_size = 1;
    }
    TakeStep(&prog.kkt_system_manager_, newton_step_parameters);
    END_TIMER

    const double d_2 = std::sqrt(std::fabs(info.normsqrd));
    const double d_inf = std::fabs(info.norminfd);

    REPORT(mu);
    REPORT(d_2);
    REPORT(d_inf);

    prog.stats.num_iter = i + 1;
    prog.stats.sqrt_inv_mu[i] = newton_step_parameters.inv_sqrt_mu;
#if CONEX_VERBOSE
    std::cout << std::endl;
#endif
  }

  if (config.prepare_dual_variables) {
    DenseMatrix y2;
    solver->Assemble(&AW, &AQc);
    solver->Factor();
    DenseMatrix bres = newton_step_parameters.inv_sqrt_mu * b - 1 * AW;
    y2 = solver->Solve(bres);

    newton_step_parameters.affine = true;
    newton_step_parameters.e_weight = 0;
    newton_step_parameters.c_weight = 0;
    Ref y2map(y2.data(), y2.rows(), y2.cols());
    StepInfo info;
    PrepareStep(&prog.kkt_system_manager_, newton_step_parameters, y2map,
                &info);
  }
  y /= newton_step_parameters.inv_sqrt_mu;
  yout = y.topRows(m);
  solved = 1;
  PRINTSTATUS("Solved.");
  return solved;
}

DenseMatrix GetFeasibleObjective(Program* prg) {
  auto& prog = *prg;
  Initialize(prog, SolverConfiguration());
  Solver solver(prog.kkt_system_manager_.cliques,
                prog.kkt_system_manager_.dual_vars);
  std::vector<KKT_SystemAssembler> kkt;
  std::list<LinearKKTAssembler> kkt_;
  for (auto& c : prog.kkt_system_manager_.eqs) {
    kkt_.push_back(LinearKKTAssembler());
    kkt_.back().workspace_ = &c.constraint;
    kkt.push_back(&kkt_.back());
  }
  solver.Bind(&kkt);

  Eigen::VectorXd AW(prog.kkt_system_manager_.SizeOfKKTSystem());
  Eigen::VectorXd AQc(prog.kkt_system_manager_.SizeOfKKTSystem());
  solver.Assemble(&AW, &AQc);

  return .5 * AW;
}

}  // namespace conex
