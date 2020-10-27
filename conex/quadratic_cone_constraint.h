#include "workspace_soc.h"
#include "newton_step.h"

class QuadraticConstraint {
  using StorageType = DenseMatrix;
 public:

  template<typename T>
  QuadraticConstraint(const DenseMatrix& Q, const T& constraint_matrix, const T& constraint_affine) : 
      workspace_( constraint_matrix.rows() - 1  ),
      Q_(Q),
      constraint_matrix_(constraint_matrix),
      constraint_affine_(constraint_affine) {
        assert(constraint_matrix_.rows() == constraint_affine_.rows());
        assert(constraint_matrix_.rows() == Q_.rows() + 1);
  }

  WorkspaceSOC* workspace() { return &workspace_; }

  int number_of_variables() { return constraint_matrix_.cols(); }
  friend int Rank(const QuadraticConstraint& o) { return 2; };
  friend void SetIdentity(QuadraticConstraint* o) { o->workspace_.W0 = 1;  o->workspace_.W1.setZero(); }
  friend void TakeStep(QuadraticConstraint* o, const StepOptions& opt, const Ref& y, StepInfo* data);
  friend void GetMuSelectionParameters(QuadraticConstraint* o,  const Ref& y, MuSelectionParameters* p);
  friend void ConstructSchurComplementSystem(QuadraticConstraint* o, bool initialize, 
                                             SchurComplementSystem* sys);
 private:
  void ComputeNegativeSlack(double inv_sqrt_mu, const Ref& y, Ref* minus_s);
  void GeodesicUpdate(const Ref& S, StepInfo* data);
  void AffineUpdate(const Ref& S);

  WorkspaceSOC workspace_;
  const DenseMatrix Q_;
  const DenseMatrix constraint_matrix_;
  const DenseMatrix constraint_affine_;
};





