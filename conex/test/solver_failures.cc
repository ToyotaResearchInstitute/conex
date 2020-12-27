#include "conex/cone_program.h"
#include "conex/equality_constraint.h"
#include "conex/linear_constraint.h"
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
}  // namespace conex

int main() { conex::EqualityConstraintFailingLDLT(); }
