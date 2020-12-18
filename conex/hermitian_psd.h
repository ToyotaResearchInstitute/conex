#pragma once
#include <Eigen/Dense>

#include "conex/jordan_matrix_algebra.h"
#include "conex/newton_step.h"
#include "conex/workspace.h"

namespace conex {

struct WorkspaceDenseHermitian {
  WorkspaceDenseHermitian(int n) : n_(n) {}
  WorkspaceDenseHermitian(int n, double* data)
      : W(data, n, n),
        temp_1(data + get_size_aligned(n * n), n, n),
        temp_2(data + get_size_aligned(n * n), n, n) {}
  static constexpr int size_of(int n) { return 3 * (get_size_aligned(n * n)); }

  friend int SizeOf(const WorkspaceDenseHermitian& o) { return size_of(o.n_); }

  friend void Initialize(WorkspaceDenseHermitian* o, double* data) {
    using Map = Eigen::Map<DenseMatrix, Eigen::Aligned>;
    int n = o->n_;
    new (&o->W) Map(data, n, n);
    new (&o->temp_1) Map(data + 1 * get_size_aligned(n * n), n, n);
    new (&o->temp_2) Map(data + 2 * get_size_aligned(n * n), n, n);
  }

  friend void print(const WorkspaceDenseHermitian& o) {
    DUMP(o.W);
    DUMP(o.temp_1);
    DUMP(o.temp_2);
  }

  Eigen::Map<DenseMatrix, Eigen::Aligned> W{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> temp_1{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> temp_2{NULL, 0, 0};
  int n_;
};

template <typename T = Real>
class HermitianPsdConstraint {
 public:
  using Matrix = typename T::Matrix;

  HermitianPsdConstraint(int n) : rank_(n), workspace_(n) {}

  HermitianPsdConstraint(int n, const std::vector<Matrix>& a, const Matrix& c)
      : rank_(n),
        workspace_(n),
        constraint_matrices_(a),
        constraint_affine_(c) {}

  WorkspaceDenseHermitian* workspace() { return &workspace_; }
  friend void SetIdentity(HermitianPsdConstraint* o) {
    o->W = T::Identity(o->rank_);
  }
  friend int Rank(const HermitianPsdConstraint& o) { return o.rank_; };

  template <typename H>
  friend void GetMuSelectionParameters(HermitianPsdConstraint<H>* o,
                                       const Ref& y, MuSelectionParameters* p);

  int number_of_variables() { return constraint_matrices_.size(); }

  template <typename H>
  friend void TakeStep(HermitianPsdConstraint<H>* o, const StepOptions& opt,
                       const Ref& y, StepInfo*);

  template <typename H>
  friend void ConstructSchurComplementSystem(HermitianPsdConstraint<H>* o,
                                             bool initialize,
                                             SchurComplementSystem* sys);

  template <typename H>
  friend bool UpdateLinearOperator(HermitianPsdConstraint<H>* o, double val,
                                   int var, int r, int c, int dim);

  template <typename H>
  friend bool UpdateAffineTerm(HermitianPsdConstraint<H>* o, double val, int r,
                               int c, int dim);

 private:
  int rank_;
  WorkspaceDenseHermitian workspace_;
  std::vector<Matrix> constraint_matrices_;
  Matrix constraint_affine_;
  Matrix W;

  double EvalDualConstraint(int j, const Matrix& W) {
    return T::TraceInnerProduct(constraint_matrices_.at(j), W);
  }
  double EvalDualObjective(const Matrix& W) {
    return T::TraceInnerProduct(constraint_affine_, W);
  }

  void ComputeNegativeSlack(double k, const Ref& y, Matrix* S) {
    *S = T::ScalarMultiply(constraint_affine_, -k);
    for (unsigned int i = 0; i < constraint_matrices_.size(); i++) {
      *S = T::Add(*S, T::ScalarMultiply(constraint_matrices_.at(i), y(i)));
    }
  }
};

using RealLMIConstraint = HermitianPsdConstraint<Real>;
using ComplexLMIConstraint = HermitianPsdConstraint<Complex>;
using QuaternicLMIConstraint = HermitianPsdConstraint<Quaternions>;
using OctonicLMIConstraint = HermitianPsdConstraint<Octonions>;

}  // namespace conex
