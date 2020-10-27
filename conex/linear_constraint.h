#include "conex/constraint.h"
#include "conex/newton_step.h"
struct WorkspaceLinear {
  WorkspaceLinear(int n, int num_vars) : n_(n), num_vars_(num_vars) {} 

  static constexpr int size_of(int n, int num_vars) { return 3*(get_size_aligned(n)) + get_size_aligned(n*num_vars);  }

  friend int SizeOf(const WorkspaceLinear& o) {
    return size_of(o.n_, o.num_vars_);
  }

  friend void Initialize(WorkspaceLinear* o, double *data) {
    using Map = Eigen::Map<DenseMatrix, Eigen::Aligned>;
     int n = o->n_;
     new (&o->W)  Map(data, n, 1);
     new (&o->temp_1) Map(data + 1*get_size_aligned(n),  n, 1);
     new (&o->temp_2) Map(data + 2*get_size_aligned(n),  n, 1);
     new (&o->weighted_constraints) Map(data + 3*get_size_aligned(n),  n,  o->num_vars_);
  }

  friend void print(const WorkspaceLinear& o) {
    DUMP(o.W);
    DUMP(o.temp_1);
    DUMP(o.temp_2);
  }

  Eigen::Map<DenseMatrix, Eigen::Aligned> W{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> temp_1{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> temp_2{NULL, 0, 0};
  Eigen::Map<DenseMatrix, Eigen::Aligned> weighted_constraints{NULL, 0, 0};
  int n_;
  int num_vars_;
};

class LinearConstraint {
  using StorageType = DenseMatrix;
 public:

  template<typename T>
  LinearConstraint(int n, T* constraint_matrix, T* constraint_affine) : 
      workspace_(n, constraint_matrix->cols()),
      constraint_matrix_(*constraint_matrix),
      constraint_affine_(*constraint_affine) { }

  LinearConstraint(int n, int m, const double* constraint_matrix, const double* constraint_affine) : 
      workspace_(n, m),
      constraint_matrix_( Eigen::Map<const DenseMatrix>(constraint_matrix, n, m)),
      constraint_affine_( Eigen::Map<const DenseMatrix>(constraint_affine, n, 1)) { }

  WorkspaceLinear* workspace() { return &workspace_; }

  friend int Rank(const LinearConstraint& o) { return o.workspace_.n_; };
  friend void SetIdentity(LinearConstraint* o);
  friend void TakeStep(LinearConstraint* o, const StepOptions& opt, const Ref& y, StepInfo* data);
  friend void GetMuSelectionParameters(LinearConstraint* o,  const Ref& y, MuSelectionParameters* p);
  friend void ConstructSchurComplementSystem(LinearConstraint* o, bool initialize, 
                                             SchurComplementSystem* sys);

  void ComputeNegativeSlack(double inv_sqrt_mu, const Ref& y, Ref* minus_s);
  void GeodesicUpdate(const Ref& S, StepInfo* data);
  void AffineUpdate(const Ref& S);
  int number_of_variables() { return constraint_matrix_.cols(); }

  WorkspaceLinear workspace_;
  const DenseMatrix constraint_matrix_;
  const DenseMatrix constraint_affine_;
};
