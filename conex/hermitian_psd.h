#pragma once
#include <Eigen/Dense>

#include "workspace.h"
#include "eigen_decomp.h"
#include "newton_step.h"
#include "jordan_matrix_algebra.h"

struct WorkspaceDenseHermitian {
  WorkspaceDenseHermitian(int n) : n_(n) {} 
  WorkspaceDenseHermitian(int n, double *data) :  
                                 W(data,  n, n),
                                 temp_1(data + get_size_aligned(n*n),  n, n),
                                 temp_2(data + get_size_aligned(n*n),  n, n)
  {}
  static constexpr int size_of(int n) { return 3*(get_size_aligned(n*n));  }

  friend int SizeOf(const WorkspaceDenseHermitian& o) {
    return size_of(o.n_);
  }

  friend void Initialize(WorkspaceDenseHermitian* o, double *data) {
    using Map = Eigen::Map<DenseMatrix, Eigen::Aligned>;
     int n = o->n_;
     new (&o->W)  Map(data, n, n);
     new (&o->temp_1) Map(data + 1*get_size_aligned(n*n),  n, n);
     new (&o->temp_2) Map(data + 2*get_size_aligned(n*n),  n, n);
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


template<typename T = Real>
class HermitianPsdConstraint {
 public:
  using Matrix = typename T::Matrix;

  HermitianPsdConstraint(int n, const std::vector<Matrix>& a, const Matrix& c) 
      : workspace_(n), constraint_matrices_(a), constraint_affine_(c) {}

  Matrix GeodesicUpdate(const Matrix& w, const Matrix& s) {
    return Geodesic<T>(w, s);
  }

  WorkspaceDenseHermitian* workspace() { return &workspace_; }
  friend void SetIdentity(HermitianPsdConstraint* o) { o->W = T::Identity(); }
  friend int Rank(const HermitianPsdConstraint& o) { return T::Rank(); };

  friend void GetMuSelectionParameters(HermitianPsdConstraint* o,  const Ref& y, MuSelectionParameters* p) {
   // using T = Real;
    using conex::jordan_algebra::SpectralRadius;
    typename T::Matrix minus_s;
    o->ComputeNegativeSlack(1, y, &minus_s);
   
    p->gw_lambda_max = NormInfWeighted<T>(o->W, minus_s);

    // <e, Q(w^{1/2}) s)
    p->gw_trace -= T().TraceInnerProduct(o->W, minus_s);
    p->gw_norm_squared += T().TraceInnerProduct(T().QuadRep(o->W, minus_s), minus_s);
  }

  int number_of_variables() { return constraint_matrices_.size(); }
  friend void TakeStep(HermitianPsdConstraint* o, const StepOptions& opt, const Ref& y, StepInfo*);

  friend void ConstructSchurComplementSystem(HermitianPsdConstraint<T>* o, bool initialize, SchurComplementSystem* sys) {
    auto G = &sys->G;
    auto& W = o->W; 
    int m = o->constraint_matrices_.size();
    if (initialize) {
      for (int i = 0; i < m; i++) {
        typename T::Matrix QA = T().QuadRep(W, o->constraint_matrices_.at(i));
        for (int j = i; j < m; j++) {
          (*G)(j, i) = o->EvalDualConstraint(j, QA);
        }

        sys->AW(i, 0)  = o->EvalDualConstraint(i, W);
        sys->AQc(i, 0) = o->EvalDualObjective(QA);
      }
    } else {
      for (int i = 0; i < m; i++) {
        typename T::Matrix QA = T().QuadRep(W, o->constraint_matrices_.at(i));
        for (int j = i; j < m; j++) {
          (*G)(j, i) += o->EvalDualConstraint(j, QA);
        }

        sys->AW(i, 0)  += o->EvalDualConstraint(i, W);
        sys->AQc(i, 0) += o->EvalDualObjective(QA);
      }
    }
  }

 private:
  WorkspaceDenseHermitian workspace_;
  std::vector<Matrix> constraint_matrices_;
  Matrix constraint_affine_;
  Matrix W;


  double EvalDualConstraint(int j, const Matrix& W) {
    return T().TraceInnerProduct(constraint_matrices_.at(j), W);
  }
  double EvalDualObjective(const Matrix& W) {
    return T().TraceInnerProduct(constraint_affine_, W);
  }

  void ComputeNegativeSlack(double k, const Ref& y, Matrix* S) {
    *S = T::ScalarMult(constraint_affine_, -k);
    for (unsigned int i = 0; i < constraint_matrices_.size(); i++) {
      *S = T().MatrixAdd(*S, T::ScalarMult(constraint_matrices_.at(i), y(i)));
    }
  }
};


using RealLMIConstraint = HermitianPsdConstraint<Real>;
using ComplexLMIConstraint = HermitianPsdConstraint<Complex>;
using QuaternicLMIConstraint = HermitianPsdConstraint<Quaternions>;
using OctonicLMIConstraint = HermitianPsdConstraint<Octonions>;
