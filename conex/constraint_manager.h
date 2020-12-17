#pragma once

#include <any>
#include <list>
#include "conex/equality_constraint.h"
#include "conex/kkt_system_assembler.h"

namespace conex {

template <typename Container>
class ConstraintManager {
 public:
  ConstraintManager(int max_number_of_variables)
      : max_number_of_variables_(max_number_of_variables),
        dual_variable_start_(max_number_of_variables_) {}

  ConstraintManager(){};

  void SetNumberOfVariables(int N) {
    max_number_of_variables_ = N;
    dual_variable_start_ = N;
  }

  int SizeOfKKTSystem() {
    int num_dual_vars = 0;
    for (auto dv : dual_vars) {
      num_dual_vars += dv.size();
    }
    return max_number_of_variables_ + num_dual_vars;
  };

  template <typename T>
  void AddConstraint(T&& x) {
    std::vector<int> clique(max_number_of_variables_);
    for (size_t i = 0; i < clique.size(); i++) {
      clique[i] = i;
    }
    AddConstraint(x, clique);
  }

  template <typename T>
  void AddConstraint(T&& x, const std::vector<int>& variables) {
    eqs.emplace_back(x);
    cliques.push_back(variables);
    dual_vars.push_back({});
  }

  void AddConstraint(EqualityConstraints&& x,
                     const std::vector<int>& variables) {
    eqs.emplace_back(x);
    cliques.push_back(variables);
    const int m = x.SizeOfDualVariable();
    dual_vars.push_back({});
    for (int i = 0; i < m; i++) {
      cliques.back().push_back(i + dual_variable_start_);
      dual_vars.back().push_back(i + dual_variable_start_);
    }
    dual_variable_start_ += m;
  }

  // Use a list so that we do not trigger reallocations.
  std::list<Container> eqs;
  std::vector<std::vector<int>> cliques;
  std::vector<std::vector<int>> dual_vars;

 private:
  int max_number_of_variables_;
  int dual_variable_start_;
};

}  // namespace conex
