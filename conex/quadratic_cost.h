#pragma once
#include "conex/cone_program.h"
#include "conex/quadratic_cone_constraint.h"

namespace conex {
inline void AddQuadraticCost(conex::Program* conex_prog,
                             const Eigen::MatrixXd& Qi,
                             const std::vector<int>& z, int epigraph) {
  // Set inner-product matrix Q of Lorentz cone L.
  Eigen::MatrixXd Q(z.size() + 1, z.size() + 1);
  Q.setZero();
  Q(0, 0) = 1;
  Q.bottomRightCorner(z.size(), z.size()) = Qi;

  // Build (A, b) satisfying b - A(x, t) \in L <=> t >= 1/2 x^T Q x.
  int num_vars = z.size();
  Eigen::MatrixXd Ai(num_vars + 2, num_vars + 1);
  Eigen::MatrixXd b(num_vars + 2, 1);
  Ai.setZero();
  b.setZero();
  Ai.topRightCorner(2, 1) << -0.5, -0.5;
  Ai.bottomLeftCorner(z.size(), z.size()) =
      Eigen::MatrixXd::Identity(num_vars, num_vars);
  b(0) = 1;
  b(1) = -1;

  // (.5 t+1)^2 >= (.5t-1)^2 + x^T Q x.
  // .25 t^2 + t + 1  >= .25 t^2 - t + 1 + x^T Q x
  // => 2t >= x^T Q x.
  auto z_indices = z;
  z_indices.push_back(epigraph);
  conex_prog->AddConstraint(conex::QuadraticConstraint(Q, Ai, b), z_indices);
}
}  // namespace conex
