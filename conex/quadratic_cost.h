#pragma once
#include "conex/cone_program.h"

namespace conex {

void AddQuadraticCost(conex::Program* conex_prog, const Eigen::MatrixXd& Qi,
                      const std::vector<int>& z);

}  // namespace conex
