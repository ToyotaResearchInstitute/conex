#include "conex/cone_program.h"

#include <vector>

#include "conex/newton_step.h"
#include "conex/divergence.h"

double CalcMinMu(double lambda_max, double, MuSelectionParameters* p) {
  const double kMaxNormInfD = p->limit;
  double inv_sqrt_mu = (1.0 + kMaxNormInfD)  / (lambda_max + 1e-12);
  if (inv_sqrt_mu < 1e-3) {
    inv_sqrt_mu = 1e-3;
  }
  return inv_sqrt_mu;
}

template<typename T>
void SetIdentity(std::vector<T>* c) {
  for (auto& ci : *c)  {
    SetIdentity(&ci);
  }
}

template<typename T>
void TakeStep(std::vector<T>* c, const StepOptions& opt, const Ref& y, StepInfo* info) {
  StepInfo info_i;
  info->normsqrd = 0;
  info->norminfd = -1;
  for (auto& ci : *c)  {
    TakeStep(&ci, opt, y, &info_i);
    if (info_i.norminfd > info->norminfd) {
      info->norminfd = info_i.norminfd;
    }
    info->normsqrd += info_i.normsqrd;
  }
}

template<typename T>
void MinMu(std::vector<T>* c, const Ref&y, MuSelectionParameters* p) {
  p->gw_norm_squared = 0;
  p->gw_lambda_max = -1000;
  for (auto& ci : *c)  {
    MinMu(&ci,  y, p);
  }
}

template<typename T>
int Rank(const std::vector<T>& c) {
  int rank = 0;
  for (const auto& ci : c)  {
    rank += Rank(ci);
  }
  return rank;
}

template<typename T>
void ConstructSchurComplementSystem(std::vector<T>* c, bool initialize, SchurComplementSystem* sys) {
  bool init = initialize;
  for (auto& ci : *c)  {
    ConstructSchurComplementSystem(&ci, init, sys);
    init = false;
  }
}


bool Solve(const DenseMatrix& b, Program& prog,
           const SolverConfiguration& config,
           double* primal_variable) {

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

  prog.InitializeWorkspace();

  SetIdentity(&constraints);

  Eigen::MatrixXd ydata(m, 1);
  Eigen::Map<DenseMatrix> yout(primal_variable, m, 1);
  Ref y(ydata.data(), m, 1);

  double inv_sqrt_mu_max = config.inv_sqrt_mu_max;
  int max_iter = 1;
  int iter_cnt = 0;

  StepOptions opt;
  opt.affine = 0;
  StepInfo info;
  IterationStats stats;
  opt.inv_sqrt_mu = 0;
  opt.affine = false;

  int rankK = Rank(constraints);

  bool error_converging = false;

  for (int i = 0; i < max_iter; i++) {
    MuSelectionParameters mu_param;

    iter_cnt++;
    if (iter_cnt > config.max_iter) {
      break;
    }

    bool init = true;
    ConstructSchurComplementSystem(&constraints, init, &sys);

    Eigen::LLT<Eigen::Ref<DenseMatrix>> llt(sys.G);
    if (llt.info() != Eigen::Success) {
      PRINTSTATUS("LLT FAILURE.");
      return false;
      break;
    }

    double div_ub  = 0;
    if ((opt.inv_sqrt_mu < inv_sqrt_mu_max) || (!error_converging)) {
      mu_param.limit = config.dinf_limit;
      y = sys.AQc - b;
      llt.solveInPlace(y);
      MinMu(&constraints,  y, &mu_param);
      
      // CalcMinMu(mu_param.gw_lambda_max, mu_param.gw_lambda_min, &mu_param);
      double divergence_upper_bound = 1;
      opt.inv_sqrt_mu = DivergenceUpperBoundInverse(divergence_upper_bound * rankK,
                                                    mu_param.gw_norm_squared,
                                                    mu_param.gw_lambda_max,
                                                    mu_param.gw_trace,
                                                    rankK);


      double normsqrd = opt.inv_sqrt_mu * opt.inv_sqrt_mu *  mu_param.gw_norm_squared +
                     -2*opt.inv_sqrt_mu * mu_param.gw_trace  + rankK;

      div_ub = normsqrd/(2 - opt.inv_sqrt_mu * mu_param.gw_lambda_max); 

      if (i > config.max_iterations - config.final_centering_steps) {
        inv_sqrt_mu_max = opt.inv_sqrt_mu;
      }
      
      max_iter = i + config.final_centering_steps;

      if (normsqrd > config.divergence_threshold * rankK) {
        if (i > 3) {
          solved = 0;
          PRINTSTATUS("Infeasible Or Unbounded.");
          return solved;
        }
      } else {
        error_converging = true;
      }
    } 

    double mu = 1.0/(opt.inv_sqrt_mu); mu *= mu;

    y = opt.inv_sqrt_mu*(b + sys.AQc) - 2 * sys.AW;
    llt.solveInPlace(y);
    opt.e_weight = 1;
    opt.c_weight = opt.inv_sqrt_mu;
    TakeStep(&constraints, opt, y, &info);

    const double d_2 = std::fabs(std::sqrt(info.normsqrd));
    const double d_inf = std::fabs(info.norminfd);

    if (i < 10) {
      std::cout << "i:  " << i << ", ";
    } else {
      std::cout << "i: " << i << ", ";
    }
    REPORT(mu);
    REPORT(div_ub);
    REPORT(d_2);
    REPORT(d_inf);

    prog.stats.num_iter = iter_cnt;
    prog.stats.sqrt_inv_mu[iter_cnt - 1] = opt.inv_sqrt_mu;
    std::cout << "\n";
  }

  if (config.prepare_dual_variables) {
    DenseMatrix y2;
    bool iter_refine = false;
    if (iter_refine) {
      ConstructSchurComplementSystem(&constraints, true, &sys);
      Eigen::LLT<Eigen::Ref<DenseMatrix>> llt(sys.G);
      DenseMatrix L = llt.matrixL();
      DenseMatrix bres = opt.inv_sqrt_mu*b - 1 * sys.AW;
      y2  = bres*0;
      for (int i = 0; i < 1; i++) {
        y2 += llt.solve(bres - L*L.transpose()*y2);
      }
    } else {
      ConstructSchurComplementSystem(&constraints, true, &sys);
      Eigen::LLT<Eigen::Ref<DenseMatrix>> llt(sys.G);
      DenseMatrix bres = opt.inv_sqrt_mu*b  - 1 * sys.AW;
      y2 = llt.solve(bres);
    }

    opt.affine = true;
    opt.e_weight = 0;
    opt.c_weight = 0;
    Ref y2map(y2.data(), y2.rows(), y2.cols());
    TakeStep(&constraints, opt, y2map, &info);
  }
  y /= opt.inv_sqrt_mu;
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
  return .5*sys.AW;
}
