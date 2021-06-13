#include "conex/cone_program.h"
#include "conex/kkt_solver.h"

#include <vector>

#include "conex/divergence.h"
#include "conex/newton_step.h"

namespace conex {

double CalcMinMu(double lambda_max, double, WeightedSlackEigenvalues* p) {
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

void GetWeightedSlackEigenvalues(ConstraintManager<Container>* constraints,
                                 const Ref& y, WeightedSlackEigenvalues* p) {
  p->frobenius_norm_squared = 0;
  p->trace = 0;
  p->lambda_max = -30000;
  p->lambda_min = 30000;
  int i = 0;
  for (auto& ci : constraints->eqs) {
    auto ysegment = Vars(y, constraints->cliques.at(i));
    Eigen::Map<Eigen::MatrixXd, Eigen::Aligned> z(ysegment.data(),
                                                  ysegment.size(), 1);
    WeightedSlackEigenvalues temp;
    GetWeightedSlackEigenvalues(&ci.constraint, z, &temp);

    if (p->lambda_max < temp.lambda_max) {
      p->lambda_max = temp.lambda_max;
    }
    if (p->lambda_min > temp.lambda_min) {
      p->lambda_min = temp.lambda_min;
    }
    p->frobenius_norm_squared += temp.frobenius_norm_squared;
    p->trace += temp.trace;

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
    prog.stats = std::make_unique<WorkspaceStats>(config.max_iterations);
    auto& solver = prog.solver;
    auto& kkt = prog.kkt;
    prog.InitializeWorkspace();
    if (config.initialization_mode == 0) {
      SetIdentity(&prog.constraints);
    }

    START_TIMER(Sparsity Analysis);
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

double MinimizeNormInf(WeightedSlackEigenvalues& p) {
  double y = -1;
  if (p.lambda_min + p.lambda_max > 0) {
    y = 2.0 / (p.lambda_min + p.lambda_max);
  }
  return y;
}

double ComputeMuFromDivergence(ConstraintManager<Container>& constraints,
                               std::unique_ptr<Solver>& solver,
                               const DenseMatrix& AQc, const DenseMatrix& b,
                               const SolverConfiguration& config, int rankK,
                               Ref* workspace_y) {
  WeightedSlackEigenvalues mu_param;
  *workspace_y = AQc - b;
  solver->SolveInPlace(workspace_y);
  GetWeightedSlackEigenvalues(&constraints, *workspace_y, &mu_param);
  mu_param.rank = rankK;

  double divergence_bound = config.divergence_upper_bound * rankK;

  double inv_sqrt_mu = 0;
  inv_sqrt_mu = DivergenceUpperBoundInverse(divergence_bound, mu_param);

  if (inv_sqrt_mu == -1) {
    inv_sqrt_mu = MinimizeNormInf(mu_param);
  }

  if (inv_sqrt_mu < 0 && mu_param.trace > 1e-12) {
    // If inverse evaluation has failed, choose mu that satisfies norm bound.
    double kstar = mu_param.trace / mu_param.frobenius_norm_squared;
    double norm_bound = 1.5 * (mu_param.frobenius_norm_squared * kstar * kstar -
                               2 * mu_param.trace * kstar + rankK);
    if (norm_bound > rankK * .7) {
      norm_bound = rankK * .7;
    }

    double a = mu_param.frobenius_norm_squared;
    double b = -2 * mu_param.trace;
    double c = rankK - norm_bound;
    if (b * b - 4 * a * c < 0) {
      inv_sqrt_mu = mu_param.trace / mu_param.frobenius_norm_squared;
    } else {
      inv_sqrt_mu = (-b + std::sqrt(b * b - 4 * a * c)) / (2 * a);
    }
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
  prog.status_.solved = 0;
  prog.status_.primal_infeasible = 0;
  prog.status_.dual_infeasible = 0;

#if CONEX_VERBOSE
  std::cout.precision(2);
  std::cout << std::scientific;
  std::cout << "Starting the Conex optimizer...\n";
#endif

  CONEX_DEMAND(prog.GetNumberOfVariables() == bin.rows(),
               "Cost vector dimension does not equal number of variables");

  int m = bin.rows();
  // Empty program
  if (prog.NumberOfConstraints() == 0) {
    Eigen::Map<DenseMatrix> ynan(primal_variable, m, 1);
    prog.status_.solved = 0;
    ynan.array() = bin.array() * std::numeric_limits<double>::infinity();
    return prog.status_.solved;
  }

  Initialize(prog, config);
  std::cout << "\n";

  Eigen::MatrixXd ydata(prog.kkt_system_manager_.SizeOfKKTSystem(), 1);
  Eigen::Map<DenseMatrix> yout(primal_variable, m, 1);
  Ref y(ydata.data(), prog.kkt_system_manager_.SizeOfKKTSystem(), 1);

  double inv_sqrt_mu_max = config.inv_sqrt_mu_max;
  double cw = 1;
  double by = -1;

  StepOptions newton_step_parameters;
  newton_step_parameters.affine = 0;
  IterationStats stats;
  newton_step_parameters.inv_sqrt_mu = 0;
  newton_step_parameters.affine = false;

  int rankK = Rank(constraints);
  int centering_steps = 0;
  bool warmstart_aborted = false;
  double inner_product_of_c_and_w;

  Eigen::VectorXd AW(prog.kkt_system_manager_.SizeOfKKTSystem());
  Eigen::VectorXd AQc(prog.kkt_system_manager_.SizeOfKKTSystem());
  Eigen::VectorXd b(prog.kkt_system_manager_.SizeOfKKTSystem());
  b.setZero();
  b.head(m) << bin;

  int initial_centering_steps = config.initial_centering_steps_coldstart;
  int initial_centering = 1;

  if (config.initialization_mode) {
    PRINTSTATUS("Warmstarting...");
    initial_centering_steps = config.initial_centering_steps_warmstart;
  }

  for (int i = 0; i < config.max_iterations; i++) {
    if (i >= initial_centering_steps) {
      initial_centering = 0;
    }

#if CONEX_VERBOSE
    if (i < 10) {
      std::cout << "i:  " << i << ", ";
    } else {
      std::cout << "i: " << i << ", ";
    }
#endif
    bool final_centering =
        (newton_step_parameters.inv_sqrt_mu >= inv_sqrt_mu_max) ||
        i >= (config.max_iterations - config.final_centering_steps);
    bool update_mu = (i == 0) || !(initial_centering || final_centering) ||
                     warmstart_aborted;
    warmstart_aborted = false;

    if (final_centering) {
      if (centering_steps >= config.final_centering_steps) {
        break;
      }
    }

    START_TIMER(Assemble)
    solver->Assemble(&AW, &AQc, &inner_product_of_c_and_w);
    END_TIMER

    START_TIMER(Factor)
    if (!solver->Factor()) {
      solver->Assemble(&AW, &AQc, &inner_product_of_c_and_w);
      solver->Factor();
      if (i == 0 && config.initialization_mode) {
        PRINTSTATUS("Aborting warmstart...");
        SetIdentity(&prog.constraints);
        warmstart_aborted = true;
        continue;
      }
      prog.status_.solved = 0;
      PRINTSTATUS("Factorization failed.");
      return prog.status_.solved;
    }
    END_TIMER

    if (update_mu) {
      double temp = ComputeMuFromDivergence(prog.kkt_system_manager_, solver,
                                            AQc, b, config, rankK, &y);
      if (temp > 0) {
        newton_step_parameters.inv_sqrt_mu = temp;
      } else {
        newton_step_parameters.inv_sqrt_mu *= .5;
      }
    } else {
      if (initial_centering == 0) {
        centering_steps++;
      }
    }

    const double max = config.inv_sqrt_mu_max;
    const double min = std::sqrt(1.0 / (1e-15 + config.maximum_mu));
    ApplyLimits(&newton_step_parameters.inv_sqrt_mu, min, max);

    double mu = 1.0 / (newton_step_parameters.inv_sqrt_mu);
    mu *= mu;

    y = newton_step_parameters.inv_sqrt_mu * (b + AQc) - 2 * AW;
    START_TIMER(Solve)
    solver->SolveInPlace(&y);
    END_TIMER

    newton_step_parameters.e_weight = 1;
    newton_step_parameters.c_weight = newton_step_parameters.inv_sqrt_mu;

    StepInfo info;
    START_TIMER(Update)
    PrepareStep(&prog.kkt_system_manager_, newton_step_parameters, y, &info);
    newton_step_parameters.step_size = 2.0 / (info.norminfd * info.norminfd);
    if (newton_step_parameters.step_size > 1) {
      newton_step_parameters.step_size = 1;
    }

    if (i == 0 && config.initialization_mode &&
        info.norminfd >= config.warmstart_abort_threshold) {
      PRINTSTATUS("Aborting warmstart...");
      SetIdentity(&prog.constraints);
      warmstart_aborted = true;
    } else {
      TakeStep(&prog.kkt_system_manager_, newton_step_parameters);
    }
    END_TIMER

    const double d_2 = std::sqrt(std::fabs(info.normsqrd));
    const double d_inf = std::fabs(info.norminfd);

    REPORT(mu);
    REPORT(d_2);
    REPORT(d_inf);
    by = b.col(0).dot(y.col(0)) * 1.0 / newton_step_parameters.inv_sqrt_mu;
    cw = inner_product_of_c_and_w * 1.0 / newton_step_parameters.inv_sqrt_mu;

    REPORT(by);
    REPORT(cw);

    prog.stats->num_iter = i + 1;
    prog.stats->sqrt_inv_mu[i] = newton_step_parameters.inv_sqrt_mu;
#if CONEX_VERBOSE
    std::cout << std::endl;
#endif

    if (final_centering ||
        newton_step_parameters.inv_sqrt_mu >= inv_sqrt_mu_max) {
      if (d_inf < config.final_centering_tolerance) {
        break;
      }
    }
  }

  yout = y.topRows(m);

  double mu = 1.0 / (newton_step_parameters.inv_sqrt_mu);
  mu *= mu;
  if (mu > config.infeasibility_threshold) {
    PRINTSTATUS("Infeasible Or Unbounded!!.");
    prog.status_.solved = 0;
    prog.status_.primal_infeasible =
        cw * newton_step_parameters.inv_sqrt_mu <= -.5;
    prog.status_.dual_infeasible =
        by * newton_step_parameters.inv_sqrt_mu >= .5;
  } else {
    PRINTSTATUS("Solved.");
    prog.status_.solved = 1;
  }

  if (config.prepare_dual_variables) {
    DenseMatrix y2;
    double cost_w;
    solver->Assemble(&AW, &AQc, &cost_w);
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

  if (prog.status_.solved) {
    yout /= newton_step_parameters.inv_sqrt_mu;
  }
  return prog.status_.solved;
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
  double inner_product_of_c_and_w;
  solver.Assemble(&AW, &AQc, &inner_product_of_c_and_w);

  return .5 * AW;
}

}  // namespace conex
