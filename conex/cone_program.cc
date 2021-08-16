#include "conex/cone_program.h"
#include "conex/kkt_solver.h"

#include <vector>

#include "conex/divergence.h"
#include "conex/newton_step.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
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

void GetWeightedSlackEigenvalues(ConstraintManager<Container>* constraints,
                                 const Ref& y, double c_weight,
                                 WeightedSlackEigenvalues* p) {
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
    GetWeightedSlackEigenvalues(&ci.constraint, z, c_weight, &temp);

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

    prog.sys.m_ = prog.kkt_system_manager_.SizeOfKKTSystem();
    prog.sys.residual_only_ = true;

    prog.InitializeWorkspace();
    if (config.initialization_mode == 0) {
      prog.stats->b_scaling() = 1;
      prog.stats->c_scaling() = 1;
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

//    y = newton_step_parameters.inv_sqrt_mu *
//            (b * b_scaling + prog.sys.AQc * c_scaling) -
//        2 * prog.sys.AW;

double ComputeMuFromLineSearch(ConstraintManager<Container>& constraints,
                               std::unique_ptr<Solver>& solver,
                               double dinf_upper_bound, const DenseMatrix& AQc,
                               double c_weight, const DenseMatrix& b,
                               const DenseMatrix& AW, Ref* y0) {
  *y0 = -2 * AW;
  solver->SolveInPlace(y0);

  VectorXd y1_data(b.rows());
  Ref y1(y1_data.data(), b.rows(), 1);
  y1 = AQc + b - 2 * AW;
  solver->SolveInPlace(&y1);
  LineSearchParameters params;
  params.c0_weight = c_weight * 0;
  params.c1_weight = c_weight * 1;
  params.dinf_upper_bound = dinf_upper_bound;
  LineSearchOutput output;

  int i = 0;
  for (auto& ci : constraints.eqs) {
    LineSearchOutput output_i;
    auto ysegment1 = Vars(*y0, constraints.cliques.at(i));
    auto ysegment2 = Vars(y1, constraints.cliques.at(i));
    Ref z1(ysegment1.data(), ysegment1.rows(), 1);
    Ref z2(ysegment2.data(), ysegment2.rows(), 1);
    bool failure = PerformLineSearch(&ci.constraint, params, z1, z2, &output_i);
    if (failure) {
      return -1;
    }
    if (output_i.lower_bound > output.lower_bound) {
      output.lower_bound = output_i.lower_bound;
    }
    if (output_i.upper_bound < output.upper_bound) {
      output.upper_bound = output_i.upper_bound;
    }
    i++;
  }
  if (output.lower_bound <= output.upper_bound) {
    return output.upper_bound;
  } else {
    return -1;
  }
}

// Finds the k that maximizes the denominator of the divergence upperbound:
//
//   max( k lambda_min, 2 - k lambda_max),
//
double MinimizeNormInf(WeightedSlackEigenvalues& p) {
  double y = -1;
  if (p.lambda_min > 0) {
    y = 2.0 / (p.lambda_min + p.lambda_max);
  }
  return y;
}
double ComputeMuFromDivergence(ConstraintManager<Container>& constraints,
                               std::unique_ptr<Solver>& solver,
                               const DenseMatrix& AQc, double c_weight,
                               const DenseMatrix& b,
                               const SolverConfiguration& config, int rankK,
                               Ref* workspace_y) {
  WeightedSlackEigenvalues mu_param;
  *workspace_y = AQc - b;
  solver->SolveInPlace(workspace_y);
  GetWeightedSlackEigenvalues(&constraints, *workspace_y, c_weight, &mu_param);
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

#ifdef EIGEN_USE_MKL_ALL
  std::cout << "...MKL Enabled\n";
#endif
#ifdef EIGEN_USE_BLAS
  std::cout << "...BLAS Enabled\n";
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
  double cx = 1;
  double by = -1;
  double kkt_error = 0;

  StepOptions newton_step_parameters;
  newton_step_parameters.affine = 0;
  IterationStats stats;
  newton_step_parameters.inv_sqrt_mu = 0;
  newton_step_parameters.affine = false;

  int rankK = Rank(constraints);
  int centering_steps = 0;
  bool warmstart_aborted = false;
  Eigen::VectorXd b(prog.kkt_system_manager_.SizeOfKKTSystem());
  b.setZero();
  b.head(m) << bin;

  int initial_centering_steps = config.initial_centering_steps_coldstart;
  int initial_centering = 1;
  auto& c_scaling = prog.stats->c_scaling();
  auto& b_scaling = prog.stats->b_scaling();

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
        (kkt_error > config.kkt_error_tolerance) ||
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
    solver->Assemble();
    AssembleSchurComplementResiduals(&prog.kkt_system_manager_, &prog.sys);
    END_TIMER

    if (i < 1 && config.initialization_mode == 0 && config.enable_rescaling) {
      b_scaling = 1.0 / (1 + b.norm());
      c_scaling = 1.0 / (1 + prog.sys.AQc.norm());
    }
    if (i < 1) {
      // The solver returns xhat = b_scaling * x and
      //                    shat = c_scaling * s
      // satisfying xhat * shat = mu * I
      //
      // This means x * s = mu/(b_scaling * c_scaling).
      // So, we rescale the target_mu by (b_scaling * c_scaling).
      double mu_target = 1.0 / (inv_sqrt_mu_max * inv_sqrt_mu_max);
      mu_target *= (b_scaling * c_scaling);
      inv_sqrt_mu_max = 1.0 / std::sqrt(mu_target);
    }

    auto M = solver->KKTMatrix();
    START_TIMER(Factor)
    if (!solver->Factor()) {
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
      double temp = -1;
      if (config.enable_line_search) {
        temp = ComputeMuFromLineSearch(prog.kkt_system_manager_, solver,
                                       config.dinf_upper_bound,
                                       prog.sys.AQc * c_scaling, c_scaling,
                                       b * b_scaling, prog.sys.AW, &y);
      }

      if (temp < 0) {
        temp = ComputeMuFromDivergence(prog.kkt_system_manager_, solver,
                                       prog.sys.AQc * c_scaling, c_scaling,
                                       b * b_scaling, config, rankK, &y);
      }

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

    const double max = inv_sqrt_mu_max;
    const double min = std::sqrt(1.0 / (1e-15 + config.maximum_mu));
    ApplyLimits(&newton_step_parameters.inv_sqrt_mu, min, max);

    y = newton_step_parameters.inv_sqrt_mu *
            (b * b_scaling + prog.sys.AQc * c_scaling) -
        2 * prog.sys.AW;
    START_TIMER(Solve)
    solver->SolveInPlace(&y);
    END_TIMER

    newton_step_parameters.e_weight = 1;
    newton_step_parameters.c_weight =
        newton_step_parameters.inv_sqrt_mu * c_scaling;

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
    by = b.col(0).dot(y.col(0)) * 1.0 /
         (newton_step_parameters.inv_sqrt_mu * c_scaling);
    // inv_sqrt_mu * <c, x> = c' Q(w^{1/2}) (e + d)
    //                      = c' Q(w^{1/2}) (e +  e + Q(w^{1/2})(Ay - k c))
    //                      = c' Q(w^{1/2}) (2e + Q(w^{1/2})(Ay - k c))
    //                      = 2 c' w + c'Q(w)(Ay - k c' Q(w) c)
    cx = 2 * prog.sys.inner_product_of_w_and_c +
         prog.sys.AQc.col(0).dot(y.col(0)) -
         newton_step_parameters.inv_sqrt_mu *
             prog.sys.inner_product_of_c_and_Qc * c_scaling;

    cx /= (newton_step_parameters.inv_sqrt_mu * b_scaling);

    double mu = 1.0 / (newton_step_parameters.inv_sqrt_mu);
    mu *= mu;

    double s_dot_x = mu * (rankK - d_2 * d_2) / (b_scaling * c_scaling);

    mu = mu / (c_scaling * b_scaling);
    REPORT(mu);
    REPORT(d_2);
    REPORT(d_inf);
    REPORT(by);
    REPORT(cx);
    kkt_error = std::fabs(cx - by - s_dot_x) / s_dot_x;
    REPORT(kkt_error);

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

  prog.status_.num_iterations = prog.stats->num_iter;
  yout = y.topRows(m);

  double mu = 1.0 / (newton_step_parameters.inv_sqrt_mu);
  mu *= mu;
  if (mu > config.infeasibility_threshold) {
    PRINTSTATUS("Infeasible Or Unbounded!!.");
    prog.status_.solved = 0;
    prog.status_.primal_infeasible =
        cx * newton_step_parameters.inv_sqrt_mu <= -.5;
    prog.status_.dual_infeasible =
        by * newton_step_parameters.inv_sqrt_mu >= .5;
  } else {
    PRINTSTATUS("Solved.");
    prog.status_.solved = 1;
  }

  if (config.prepare_dual_variables) {
    DenseMatrix y2;
    solver->Assemble();
    AssembleSchurComplementResiduals(&prog.kkt_system_manager_, &prog.sys);
    solver->Factor();
    DenseMatrix bres =
        newton_step_parameters.inv_sqrt_mu * b * b_scaling - 1 * prog.sys.AW;
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
    yout /= (newton_step_parameters.inv_sqrt_mu);
    yout /= c_scaling;
  }
  return prog.status_.solved;
}

DenseMatrix GetFeasibleObjective(Program* prg) {
  auto& prog = *prg;
  Initialize(prog, SolverConfiguration());

  Eigen::VectorXd AW(prog.kkt_system_manager_.SizeOfKKTSystem());
  Eigen::VectorXd AQc(prog.kkt_system_manager_.SizeOfKKTSystem());
  double inner_product_of_c_and_w;
  prog.solver->Assemble(&AW, &AQc, &inner_product_of_c_and_w);

  return .5 * AW;
}

}  // namespace conex
