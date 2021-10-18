#include "conex/cone_program.h"
#include "conex/newton_step.h"

namespace conex {

namespace {
struct WorkspaceQuadraticFunction {
  using DenseMatrix = Eigen::MatrixXd;

  friend int SizeOf(const WorkspaceQuadraticFunction&) { return 0; }

  friend void Initialize(WorkspaceQuadraticFunction*, double*) {}

  friend void print(const WorkspaceQuadraticFunction&) {}
  Eigen::Map<DenseMatrix, Eigen::Aligned> W{NULL, 0, 0};
};

class QuadraticFunction {
 public:
  QuadraticFunction(const Eigen::MatrixXd& A) : A_(A) {}

  int SizeOfDualVariable() { return A_.rows(); }
  Eigen::MatrixXd A_;

  friend int Rank(const QuadraticFunction&) { return 0; };
  friend void SetIdentity(QuadraticFunction*){};
  friend void PrepareStep(QuadraticFunction* o, const StepOptions&,
                          const Ref& y, StepInfo* info_i) {
    info_i->normsqrd = 0;
    info_i->norminfd = 0;
  }

  friend bool UpdateAffineTerm(QuadraticFunction* o, double val, int r, int c,
                               int dim) {
    CONEX_DEMAND(dim == 0, "Quadratic cost must be real valued matrix.");
    CONEX_DEMAND(r < o->A_.rows() && c < o->A_.cols(), "Index out of bounds");
    o->A_(r, c) = val;
    return CONEX_SUCCESS;
  }

  friend bool TakeStep(QuadraticFunction*, const StepOptions&) { return true; };

  friend void GetWeightedSlackEigenvalues(QuadraticFunction*, const Ref&,
                                          double, WeightedSlackEigenvalues*){};

  friend void ConstructSchurComplementSystem(QuadraticFunction* o,
                                             bool initialize,
                                             SchurComplementSystem* sys_) {
    auto& sys = *sys_;
    auto& A_ = o->A_;
    if (initialize) {
      sys.setZero();
      sys.G.topLeftCorner(A_.rows(), A_.cols()) = A_;
    } else {
      sys.G.topLeftCorner(A_.rows(), A_.cols()) += A_;
    }
  }

  friend bool PerformLineSearch(QuadraticFunction* o,
                                const LineSearchParameters& params,
                                const Ref& y0, const Ref& y1,
                                LineSearchOutput* output) {
    bool failure = false;
    return failure;
  }

  int number_of_variables() { return 0; }
  WorkspaceQuadraticFunction workspace_;
  WorkspaceQuadraticFunction* workspace() { return &workspace_; }
};
}  // namespace

void AddQuadraticCost(conex::Program* conex_prog, const Eigen::MatrixXd& Qi,
                      const std::vector<int>& z) {
  conex_prog->AddConstraint(QuadraticFunction{Qi}, z);
}

}  // namespace conex
