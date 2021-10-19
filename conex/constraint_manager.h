#pragma once

#include <any>
#include <list>
#include "conex/equality_constraint.h"
#include "conex/kkt_system_assembler.h"

#include "conex/error_checking_macros.h"

namespace conex {

inline int IsUnique(int N, const std::vector<int>& x) {
  Eigen::VectorXd y(N);
  y.setZero();
  for (auto& xi : x) {
    if (xi >= N) {
      return false;
    }
    y(xi)++;
    if (y(xi) > 1) {
      return false;
    }
  }
  return true;
}

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
  bool AddConstraint(T&& x) {
    std::vector<int> clique(max_number_of_variables_);
    for (size_t i = 0; i < clique.size(); i++) {
      clique[i] = i;
    }
    AddConstraint(x, clique);
    return CONEX_SUCCESS;
  }

  template <typename T>
  bool AddConstraint(T&& x, const std::vector<int>& variables) {
    if (!IsUnique(max_number_of_variables_, variables)) {
      return CONEX_FAILURE;
    }
    eqs.emplace_back(x, variables.size());
    cliques.push_back(variables);
    dual_vars.push_back({});
    return CONEX_SUCCESS;
  }

  bool AddEqualityConstraint(EqualityConstraints&& x,
                             const std::vector<int>& variables) {
    if (!IsUnique(max_number_of_variables_, variables)) {
      return CONEX_FAILURE;
    }
    const int m = x.SizeOfDualVariable();
    eqs.emplace_back(x, m + variables.size());
    cliques.push_back(variables);
    dual_vars.push_back({});
    for (int i = 0; i < m; i++) {
      cliques.back().push_back(i + dual_variable_start_);
      dual_vars.back().push_back(i + dual_variable_start_);
    }
    dual_variable_start_ += m;
    return CONEX_SUCCESS;
  }

  bool AddEqualityConstraint(EqualityConstraints&& x) {
    std::vector<int> clique(max_number_of_variables_);
    for (size_t i = 0; i < clique.size(); i++) {
      clique[i] = i;
    }
    AddEqualityConstraint(std::forward<EqualityConstraints>(x), clique);
    return CONEX_SUCCESS;
  }

  // Use a list so that we do not trigger reallocations.
  std::list<Container> eqs;
  std::vector<std::vector<int>> cliques;
  std::vector<std::vector<int>> dual_vars;

 private:
  int max_number_of_variables_ = 0;
  int dual_variable_start_ = 0;
};

template <typename Container>
void AssembleSchurComplementResiduals(ConstraintManager<Container>* kkt,
                                      SchurComplementSystem* s) {
  s->setZero();
  int i = 0;
  for (auto& ci : kkt->eqs) {
    auto* rhs_i = &ci.kkt_assembler.schur_complement_data;
    s->inner_product_of_w_and_c += rhs_i->inner_product_of_w_and_c;
    s->inner_product_of_c_and_Qc += rhs_i->inner_product_of_c_and_Qc;
    int cnt = 0;
    for (auto k : kkt->cliques.at(i)) {
      s->AW(k) += rhs_i->AW(cnt);
      s->AQc(k) += rhs_i->AQc(cnt);
      cnt++;
    }
    i++;
  }
}

}  // namespace conex
