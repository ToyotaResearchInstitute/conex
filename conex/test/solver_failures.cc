#include "conex/cone_program.h"
#include "conex/equality_constraint.h"
#include "conex/linear_constraint.h"
#include "conex/quadratic_cone_constraint.h"
#include <Eigen/Dense>

namespace conex {

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Creates KKT matrix of the form:
//  G << 1, 1,  1,
//       1, 1, -1,
//       1, -1, 0;
// which we fail to factor.
void EqualityConstraintFailingLDLT() {
  int num_vars = 2;
  int num_equalities = 1;
  int num_inequalities = 1;

  MatrixXd A = MatrixXd::Random(num_inequalities, num_vars);
  A << 1, 1;
  MatrixXd C(num_inequalities, 1);
  C.setConstant(1);
  LinearConstraint linear_inequality{A, C};

  MatrixXd eq = MatrixXd::Random(num_equalities, num_vars);
  MatrixXd eq_affine(num_equalities, 1);
  eq_affine.setZero();
  eq << 1, -1;

  Program prog(num_vars);
  prog.AddConstraint(EqualityConstraints{eq, eq_affine}, {0, 1});
  prog.AddConstraint(linear_inequality);

  VectorXd linear_cost(num_vars);
  linear_cost = A.transpose() * C;

  VectorXd solution(num_vars);
  Solve(linear_cost, prog, conex::SolverConfiguration(), solution.data());

  DUMP(eq * solution - eq_affine);
  DUMP(solution);
}

// The centering parameter moves around.
void DoBadInitialization(bool fail) {
  int num_vars = 5;
  int num_inequalities = num_vars + 10;
  int num_equalities = 1;

  MatrixXd A = MatrixXd::Random(num_inequalities, num_vars);
  MatrixXd C(num_inequalities, 1);

  VectorXd optimal_slack(num_inequalities);
  VectorXd optimal_dual(num_inequalities);
  VectorXd optimal_y(num_vars);

  optimal_slack.setConstant(1);
  optimal_dual.setConstant(1);
  int m = num_inequalities * .5;
  optimal_slack.topRows(m).setConstant(1e-3);
  optimal_dual.bottomRows(num_inequalities - m).setConstant(1e-3);

  optimal_y = Eigen::MatrixXd::Random(num_vars, 1);

  C = optimal_slack + A * optimal_y;

  LinearConstraint linear_inequality{A, C};

  Program prog(num_vars);
  MatrixXd eq = MatrixXd::Zero(num_equalities, num_vars);
  Eigen::MatrixXd Bi(1, 3);
  Bi << 1, 2, 3;
  for (int i = 0; i < num_equalities; i++) {
    std::vector<int> vars{0, i + 1, num_vars - 1};
    for (size_t j = 0; j < vars.size(); j++) {
      eq(i, vars.at(j)) = Bi(0, j);
    }
    // prog.AddConstraint(EqualityConstraints{Bi, eq.row(i) * optimal_y    },
    // vars);
  }

  MatrixXd eq_affine(num_equalities, 1);
  eq_affine = eq * optimal_y;

  prog.AddConstraint(EqualityConstraints{eq, eq_affine});
  prog.AddConstraint(linear_inequality);

  VectorXd linear_cost(num_vars);
  linear_cost = A.transpose() * optimal_dual;

  VectorXd solution(num_vars);
  auto config = conex::SolverConfiguration();
  config.final_centering_steps = 10;

  // TODO(FrankPermenter): Understand occasional poor convergence when lowered.
  config.initial_centering_steps_coldstart = 0;
  if (!fail) {
    config.initial_centering_steps_coldstart = 10;
  }
  config.max_iterations = 40;
  config.divergence_upper_bound = .5;
  Solve(linear_cost, prog, config, solution.data());
}

class MPCFailingLDLT {
 public:
  template <typename T>
  using vector = std::vector<T>;
  static constexpr int T = 3;
  static constexpr int nu = 1;
  static constexpr int nx = 2;

  auto InputVars(int i) {
    int offset = T * nx + i * nu;
    // x, u x u
    vector<int> y;
    for (int i = 0; i < nu; i++) {
      y.push_back(i + offset);
    }
    return y;
  }

  auto StateVars(int i) {
    assert(i >= 1);
    int offset = (i - 1) * nx;
    vector<int> y;
    for (int i = 0; i < nx; i++) {
      y.push_back(i + offset);
    }
    return y;
  }

  auto StageVars(int i, bool next_state) {
    vector<int> y;
    if (i > 0) {
      for (const auto& c : StateVars(i)) {
        y.push_back(c);
      }
    }

    for (const auto& c : InputVars(i)) {
      y.push_back(c);
    }

    for (const auto& c : StateVars(i + 1)) {
      y.push_back(c);
    }
    return y;
  }

  auto DynamicsConstraint(const MatrixXd& Ai, const MatrixXd& Bi, int i) {
    if (i > 0) {
      MatrixXd A(nx, 2 * nx + nu);
      A << Ai, Bi, -MatrixXd::Identity(nx, nx);
      return A;
    } else {
      MatrixXd A(nx, nx + nu);
      A << Bi, -MatrixXd::Identity(nx, nx);
      return A;
    }
  }

  void Run(bool fail) {
    int num_vars = T * (nu + nx + 2);
    int epigraph_start = T * (nu + nx);

    MatrixXd Ai = MatrixXd::Random(nx, nx);
    MatrixXd Bi = MatrixXd::Random(nx, nu);
    MatrixXd f = MatrixXd::Random(nx, 1);
    MatrixXd Hxu = MatrixXd::Random(2, nu);
    MatrixXd gxu = MatrixXd::Random(2, 1);

    conex::Program prog(num_vars);
    for (int i = 0; i < T; i++) {
      MatrixXd M(nx, nu + 2 * nx);
      prog.AddConstraint(
          conex::EqualityConstraints{DynamicsConstraint(Ai, Bi, i), f},
          StageVars(i, true));

      if (fail) {
        if (i > 0 && i < T - 1) {
          prog.AddConstraint(conex::LinearConstraint{Hxu, gxu},
                             StageVars(i, false));
        }
      }

      AddQuadraticCostEpigraph(&prog, MatrixXd::Identity(nu, nu), InputVars(i),
                               epigraph_start++);
      AddQuadraticCostEpigraph(&prog, MatrixXd::Identity(nx, nx),
                               StateVars(i + 1), epigraph_start++);
    }

    Eigen::VectorXd var(num_vars);
    VectorXd linear_cost(num_vars);
    linear_cost.setConstant(-1);
    auto config = conex::SolverConfiguration();
    config.inv_sqrt_mu_max = 1e4;
    config.final_centering_steps = 10;
    config.max_iterations = 50;
    conex::Solve(linear_cost, prog, config, var.data());
    DUMP(var);
  }
};

void LPFailSlater(int number_of_implicit_equations) {
  double distance_to_infeasible = 0;
  SolverConfiguration config;
  config.prepare_dual_variables = true;
  config.inv_sqrt_mu_max = 100;
  config.final_centering_tolerance = 1;
  config.infeasibility_threshold = 2000000;
  config.final_centering_steps = 5;

  int m = 10;
  int n1 = number_of_implicit_equations;
  int n2 = 8;
  int n = 2 * n1 + n2;
  Eigen::MatrixXd yref = DenseMatrix::Random(m, 1);

  DenseMatrix A1 = DenseMatrix::Random(n1, m);
  DenseMatrix C1 = A1 * yref;
  DenseMatrix A2 = DenseMatrix::Random(n2, m);
  DenseMatrix C2 = A2 * yref;
  C2.array() += 2;

  DenseMatrix A = DenseMatrix::Random(n, m);
  DenseMatrix C = DenseMatrix::Random(n, 1);
  A << A1, -A1, A2;

  DenseMatrix offset(n1, 1);
  offset.setConstant(distance_to_infeasible);
  C << C1, -(C1 - offset), C2;

  LinearConstraint _constraint{n, &A, &C};

  Program prog(m);
  prog.SetNumberOfVariables(m);
  prog.AddConstraint(_constraint);

  VectorXd b(2);
  VectorXd xref = VectorXd::Random(n);
  xref = xref.array().abs();
  b = A.transpose() * xref;

  DenseMatrix y(m, 1);
  Solve(b, prog, config, y.data());
}
}  // namespace conex

int main() {
  conex::EqualityConstraintFailingLDLT();
  conex::MPCFailingLDLT().Run(true /*trigger fail*/);
  srand(0);
  for (int i = 0; i < 5; i++) {
    conex::DoBadInitialization(true /*trigger fail*/);
  }
  // Triggers factorization failure
  srand(0);
  conex::LPFailSlater(1 /*num implicit eqs*/);
  // Triggers bad convergence
  srand(0);
  conex::LPFailSlater(2 /*num implicit eqs*/);
}
