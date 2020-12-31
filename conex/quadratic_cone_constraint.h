#include "newton_step.h"
#include "workspace_soc.h"

namespace conex {

class QuadraticConstraint {
  using StorageType = DenseMatrix;

 public:
  template <typename T>
  QuadraticConstraint(const DenseMatrix& Q, const T& constraint_matrix,
                      const T& constraint_affine)
      : n_(constraint_matrix.rows() - 1),
        workspace_(n_),
        Q_(Q),
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
  QuadraticConstraint(const T& constraint_matrix, const T& constraint_affine)
      : QuadraticConstraint(DenseMatrix(), constraint_matrix,
                            constraint_affine) {}

  WorkspaceSOC* workspace() { return &workspace_; }

  int number_of_variables() { return A1_.cols(); }
  friend int Rank(const QuadraticConstraint&) { return 2; };
  friend void SetIdentity(QuadraticConstraint* o) {
    o->workspace_.W0 = 1;
    o->workspace_.W1.setZero();
  }
  friend void TakeStep(QuadraticConstraint* o, const StepOptions& opt,
                       const Ref& y, StepInfo* data);
  friend void GetMuSelectionParameters(QuadraticConstraint* o, const Ref& y,
                                       MuSelectionParameters* p);
  friend void ConstructSchurComplementSystem(QuadraticConstraint* o,
                                             bool initialize,
                                             SchurComplementSystem* sys);

 private:
  void Initialize();
  void ComputeNegativeSlack(double inv_sqrt_mu, const Ref& y, Ref* minus_s);
  void GeodesicUpdate(const Ref& S, StepInfo* data);
  void AffineUpdate(const Ref& S);

  const int n_ = 0;
  WorkspaceSOC workspace_;
  const DenseMatrix Q_;

  const Eigen::VectorXd A0_;
  const Eigen::MatrixXd A1_;
  const double C0_;
  const Eigen::VectorXd C1_;
  Eigen::MatrixXd A_gram_;
};

}  // namespace conex
