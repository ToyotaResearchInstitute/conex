#pragma once
#include "cone_program.h"
#include "newton_step.h"
#include "workspace_soc.h"

namespace conex {

class QuadraticConstraintBase {
  using StorageType = DenseMatrix;

 public:
  template <typename T>
  QuadraticConstraintBase(const DenseMatrix& Q, const T& constraint_matrix,
                          const T& constraint_affine)
      : Q_(Q),
        n_(constraint_matrix.rows() - 1),
        workspace_(n_),
        A0_(constraint_matrix.row(0)),
        A1_(constraint_matrix.bottomRows(n_)),
        C0_(constraint_affine(0, 0)),
        C1_(constraint_affine.bottomRows(n_).leftCols(1)) {
    assert(constraint_matrix.rows() == constraint_affine.rows());
    assert(constraint_matrix.rows() == n_ + 1);
    assert(Q_.rows() == n_ || /*We assume Q = I*/ Q_.rows() == 0);
    assert(constraint_affine.cols() == 1);
    Initialize();
  }

  template <typename T>
  QuadraticConstraintBase(const T& constraint_matrix,
                          const T& constraint_affine)
      : QuadraticConstraintBase(DenseMatrix(), constraint_matrix,
                                constraint_affine) {}

  WorkspaceSOC* workspace() { return &workspace_; }

  int number_of_variables() { return A1_.cols(); }
  friend int Rank(const QuadraticConstraintBase&) { return 2; };
  friend void SetIdentity(QuadraticConstraintBase* o);
  friend void PrepareStep(QuadraticConstraintBase* o, const StepOptions& opt,
                          const Ref& y, StepInfo* data);

  friend bool TakeStep(QuadraticConstraintBase* o, const StepOptions& opt);
  friend void GetWeightedSlackEigenvalues(QuadraticConstraintBase* o,
                                          const Ref& y, double c_weight,
                                          WeightedSlackEigenvalues* p);
  friend void ConstructSchurComplementSystem(QuadraticConstraintBase* o,
                                             bool initialize,
                                             SchurComplementSystem* sys);

  virtual ~QuadraticConstraintBase(){};

 protected:
  virtual void Initialize();
  virtual DenseMatrix EvalAtQX(const DenseMatrix& X, DenseMatrix* QX);
  virtual DenseMatrix EvalAtQX(const DenseMatrix& X, Ref* QX);
  double EvalCQX(const DenseMatrix& X, Ref* QX);

  const DenseMatrix Q_;

 private:
  void ComputeNegativeSlack(double inv_sqrt_mu, const Ref& y, double* minus_s_0,
                            Ref* minus_s_1);
  void GeodesicUpdate(const Ref& S, StepInfo* data);
  void AffineUpdate(const Ref& S);

  const int n_ = 0;
  WorkspaceSOC workspace_;

  const Eigen::VectorXd A0_;
  const Eigen::MatrixXd A1_;
  const double C0_;
  const Eigen::VectorXd C1_;

  // TODO(FrankPermenter): Move to workspace.
  Eigen::MatrixXd A_gram_;
  Eigen::MatrixXd A_dot_x_;
};

using QuadraticConstraint = QuadraticConstraintBase;

class QuadraticEpigraph : public QuadraticConstraintBase {
 public:
  QuadraticEpigraph(const DenseMatrix& Qi);

 private:
  void Initialize() override;
  DenseMatrix EvalAtQX(const DenseMatrix& X, DenseMatrix* QX) override;
  DenseMatrix EvalAtQX(const DenseMatrix& X, Ref* QX) override;
};
inline void AddQuadraticCostEpigraph(conex::Program* conex_prog,
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
