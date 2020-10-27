#include "conex/newton_step.h"
#include "conex/workspace_soc.h"

class SOCConstraint {
  using StorageType = DenseMatrix;
 public:

  template<typename T>
  SOCConstraint(const T& constraint_matrix, const T& constraint_affine) : 
      workspace_( constraint_matrix.rows() - 1  ),
      constraint_matrix_(constraint_matrix),
      constraint_affine_(constraint_affine) {
        assert(constraint_matrix_.rows() == constraint_affine_.rows());
      }

  WorkspaceSOC* workspace() { return &workspace_; }

  int number_of_variables() { return constraint_matrix_.cols(); }
  friend int Rank(const SOCConstraint& o) { return 2; };
  friend void SetIdentity(SOCConstraint* o) { o->workspace_.W0 = 1;  o->workspace_.W1.setZero(); }
  friend void TakeStep(SOCConstraint* o, const StepOptions& opt, const Ref& y, StepInfo* data);
  friend void GetMuSelectionParameters(SOCConstraint* o,  const Ref& y, MuSelectionParameters* p);
  friend void ConstructSchurComplementSystem(SOCConstraint* o, bool initialize, 
                                             SchurComplementSystem* sys);
 private:
  void ComputeNegativeSlack(double inv_sqrt_mu, const Ref& y, Ref* minus_s);
  void GeodesicUpdate(const Ref& S, StepInfo* data);
  void AffineUpdate(const Ref& S);

  WorkspaceSOC workspace_;
  const DenseMatrix constraint_matrix_;
  const DenseMatrix constraint_affine_;
};





