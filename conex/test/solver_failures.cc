#include "conex/cone_program.h"
#include "conex/equality_constraint.h"
#include "conex/linear_constraint.h"
#include "conex/quadratic_cost.h"
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

      AddQuadraticCost(&prog, MatrixXd::Identity(nu, nu), InputVars(i),
                       epigraph_start++);
      AddQuadraticCost(&prog, MatrixXd::Identity(nx, nx), StateVars(i + 1),
                       epigraph_start++);
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

}  // namespace conex

int main() {
  conex::EqualityConstraintFailingLDLT();
  conex::MPCFailingLDLT().Run(true /*trigger fail*/);
}
