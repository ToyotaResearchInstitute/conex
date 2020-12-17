#include "conex/soc_constraint.h"
#include "conex/error_checking_macros.h"
#include "conex/newton_step.h"

namespace conex {

using EigenType = DenseMatrix;
using Real = double;

// Implements the spectral decomposition of the Spin Factor algebra.
// See
// http://rutcor.rutgers.edu/~alizadeh/CLASSES/12fallSDP/Notes/Lecture08/lec08.pdf
// or "Analysis on Symmetric Cones" by Faraut and Koranyi.
class SpectralDecompSpinFactor {
 public:
  using EssentialVectorType = DenseMatrix;

  SpectralDecompSpinFactor(int n) : n_(n), q_(n, 1) {}
  int n_;

  struct PeirceDecompType {
    PeirceDecompType(int n) : X00(n + 1), X11(n + 1), X01(n + 1) {}

    Eigen::VectorXd X00;
    Eigen::VectorXd X11;
    Eigen::VectorXd X01;
    Eigen::VectorXd Component(int i, int j) {
      if ((i == 0) && (j == 0)) {
        return X00;
      }
      if ((i + j == 1)) {
        return X01;
      }
      if ((i == 1) && (j == 1)) {
        return X11;
      }
      return EigenType();
    }
  };

  Eigen::Matrix<Real, 2, 1> Eigenvalues() const { return eigenvalues_; }
  void Compute(const Eigen::VectorXd& x) {
    assert(x.rows() == n_ + 1);
    q_ = x.col(0).tail(n_);
    norm_of_q_ = q_.norm();
    if (norm_of_q_ > 0) {
      q_ = q_ / norm_of_q_;
    }
    eigenvalues_(0) = x(0) + norm_of_q_;
    eigenvalues_(1) = x(0) - norm_of_q_;
  }

  EigenType Idempotent(int i) const { return Idempotents().col(i); }

  // Implements equations from page 7 of
  // http://rutcor.rutgers.edu/~alizadeh/CLASSES/12fallSDP/Notes/Lecture08/lec08.pdf
  DenseMatrix Idempotents() const {
    int n = n_;
    DenseMatrix idempotents(n + 1, 2);
    if (norm_of_q_ > 0) {
      idempotents.col(0) << .5, .5 * q_;
      idempotents.col(1) << .5, -.5 * q_;
    } else {
      idempotents.setZero();
      idempotents(0, 0) = .5;
      idempotents(0, 1) = .5;
    }
    return idempotents;
  }

  // The 3 Peirce components of x are the orthogonal projections of
  // x onto the following 3 subspaces:
  //
  //   S00 := span { (1,  q) }               (dim = 0)
  //   S11 := span { (1, -q) }               (dim = 0)
  //   S01 :=  (S00 + S11)^{\perp}
  //        = { (0, p) : <p, q> = 0 }
  //
  // See, e.g., Example 06 of "An Introduction to
  // Formally Real Jordan Algebras and Their Applications in Optimization" by
  // Alizadeh.
  PeirceDecompType TransformToPeirceComponents(const Eigen::VectorXd& x) const {
    PeirceDecompType peirce_decomp(x.rows());
    // Compute X00 and X11 by directly computing orthogonal projection
    // onto S00 and S11
    int size = n_;
    const double c0 = .5 * x(0);
    const double c1 = .5 * q_.dot(x.tail(size - 1));
    peirce_decomp.X00(0) = c0 + c1;
    peirce_decomp.X00.tail(size - 1) = (c0 + c1) * q_;
    peirce_decomp.X11(0) = c0 - c1;
    peirce_decomp.X11.tail(size - 1) = (c1 - c0) * q_;

    // Compute X01 by using the fact that S01 + S00 + S11 is a direct-sum
    // decomposition
    peirce_decomp.X01 = x - peirce_decomp.X00 - peirce_decomp.X11;
    return peirce_decomp;
  }

  EigenType TransformFromPeirceComponents(const PeirceDecompType& X) const {
    return X.X00 + X.X11 + X.X01;
  }

  // If we have computed the spectral decomposition of (x0, x1), then the
  // essential unit vector is x1*1/|x1|.
  auto EssentialUnitVector() const { return q_; }
  auto NormOfEssentialVector() const { return norm_of_q_; }

  // Let z = (z0, z1) have Peirce decomposition
  //      z = c0 (1, q) + c1 (1, -q)  + (0, p).
  //
  // This function returns (c0-c1) q, i.e., the essential vector of just the
  // "diagonal" Peirce components
  //        c0 (1, q) + c1 (1, -q).
  //
  // Since z1 = (c0 - c1) q + p  and  <p , q> = 0, we compute this simply by
  // projecting the essential vector z1 onto the span of q.
  EssentialVectorType EssentialVectorOfDiagonalPeirceComponents(
      const Eigen::VectorXd& z) const {
    const double inner_product = q_.tail(n_ - 1).dot(z);
    return q_ * inner_product;
  }

 private:
  Eigen::Matrix<Real, 2, 1> eigenvalues_;
  Eigen::VectorXd q_;
  Real norm_of_q_;
};

DenseMatrix QuadraticRepresentation(const Eigen::VectorXd& x,
                                    const Eigen::VectorXd& y) {
  // We use the formula from Example 11.12 of "Formally Real Jordan Algebras
  // and Their Applications to Optimization"  by Alizadeh, which states the
  // quadratic representation of x equals the linear map
  //                          2xx' - (det x) * R
  // where R is the reflection operator R = diag(1, -1, ..., -1) and det x is
  // the determinate of x = (x0, x1), i.e., det x = x0^2 - |x1|^2.
  int order = x.rows();
  double det_x = x(0) * x(0) - x.tail(order - 1).squaredNorm();
  EigenType z = det_x * y;
  z(0) *= -1;
  return (2 * x.dot(y)) * x + z;
}

DenseMatrix Sqrt(double x0, const DenseMatrix& x) {
  int n = x.rows();
  DenseMatrix z(n + 1, 1);
  z(0, 0) = x0;
  z.bottomRows(n) = x;
  SpectralDecompSpinFactor spec(n);
  spec.Compute(z);
  auto ev = spec.Eigenvalues();

  if (ev.minCoeff() < 0) {
    DUMP(ev);
    DUMP(z);
    assert(0);
  }

  DenseMatrix zsqrt = std::sqrt(ev(0, 0)) * spec.Idempotent(0) +
                      std::sqrt(ev(1, 0)) * spec.Idempotent(1);
  return zsqrt;
}

DenseMatrix Exp(double x0, const DenseMatrix& x) {
  int n = x.rows();
  DenseMatrix z(n + 1, 1);
  z(0, 0) = x0;
  z.bottomRows(n) = x;
  SpectralDecompSpinFactor spec(n);
  spec.Compute(z);
  auto ev = spec.Eigenvalues();
  DenseMatrix zsqrt = std::exp(ev(0, 0)) * spec.Idempotent(0) +
                      std::exp(ev(1, 0)) * spec.Idempotent(1);
  return zsqrt;
}

double NormInf(double x0, const DenseMatrix& x) {
  int n = x.rows();
  DenseMatrix z(n + 1, 1);
  z(0, 0) = x0;
  z.bottomRows(n) = x;
  SpectralDecompSpinFactor spec(n);
  spec.Compute(z);
  auto ev = spec.Eigenvalues();
  if (std::fabs(ev(0)) > std::fabs(ev(1))) {
    return std::fabs(ev(0));
  } else {
    return std::fabs(ev(1));
  }
}

void SOCConstraint::ComputeNegativeSlack(double inv_sqrt_mu, const Ref& y,
                                         Ref* minus_s) {
  minus_s->noalias() = (constraint_matrix_)*y;
  minus_s->noalias() -= (constraint_affine_)*inv_sqrt_mu;
}

// Combine this with TakeStep
void GetMuSelectionParameters(SOCConstraint* o, const Ref& y,
                              MuSelectionParameters* p) {
  auto* workspace = &o->workspace_;
  auto& minus_s = workspace->temp_1;
  auto& Ws = workspace->temp_2;
  o->ComputeNegativeSlack(1, y, &minus_s);

  auto wsqrt = Sqrt(o->workspace_.W0, o->workspace_.W1);
  int n = workspace->n_;
  Ws = QuadraticRepresentation(wsqrt, minus_s);

  SpectralDecompSpinFactor spec(n);
  spec.Compute(Ws);
  auto ev = spec.Eigenvalues();

  const double lamda_max = -ev.minCoeff();
  const double lamda_min = -ev.maxCoeff();

  if (p->gw_lambda_max < lamda_max) {
    p->gw_lambda_max = lamda_max;
  }
  if (p->gw_lambda_min > lamda_min) {
    p->gw_lambda_min = lamda_min;
  }
  p->gw_norm_squared += Ws.squaredNorm();
  p->gw_trace += -Ws.sum();
}

void TakeStep(SOCConstraint* o, const StepOptions& opt, const Ref& y,
              StepInfo* info) {
  auto& minus_s = o->workspace_.temp_1;
  o->ComputeNegativeSlack(opt.inv_sqrt_mu, y, &minus_s);

  // e - Q(w^{1/2})(C-A^y)
  auto wsqrt = Sqrt(o->workspace_.W0, o->workspace_.W1);
  int n = wsqrt.rows();
  auto d = QuadraticRepresentation(wsqrt, minus_s);
  d(0, 0) += 1;

  info->norminfd = NormInf(d(0, 0), d.bottomRows(n - 1));
  info->normsqrd = d.squaredNorm();

  double scale = info->norminfd * info->norminfd;
  if (scale > 2.0) {
    d = 2 * d / scale;
  }
  auto expd = Exp(d(0, 0), d.bottomRows(n - 1));
  auto wn = QuadraticRepresentation(wsqrt, expd);
  o->workspace_.W0 = wn(0, 0);
  o->workspace_.W1 = wn.bottomRows(n - 1);

  if (o->workspace_.W1.norm() > o->workspace_.W0) {
    DUMP(o->workspace_.W1.norm());
    DUMP(o->workspace_.W0);
    assert(0);
  }

  // Q(w^{1/2}) exp Q(w^{1/2}) d
  //                (w w^T - det w R) d
  //  (w w^T  - det w R) exp  (w w^T - det w R) d
  //
  //  exp(ld1) lw1  d
  //  exp(ld1) lw2  d
}

void ConstructSchurComplementSystem(SOCConstraint* o, bool initialize,
                                    SchurComplementSystem* sys) {
  int n = o->workspace_.n_;
  auto Wsqrt = Sqrt(o->workspace_.W0, o->workspace_.W1);
  DenseMatrix W(n + 1, 1);
  W(0, 0) = o->workspace_.W0;
  W.bottomRows(n) = o->workspace_.W1;

  auto G = &sys->G;

  Eigen::MatrixXd WA = o->constraint_matrix_;
  Eigen::MatrixXd WC = QuadraticRepresentation(Wsqrt, o->constraint_affine_);

  for (int i = 0; i < WA.cols(); i++) {
    WA.col(i) = QuadraticRepresentation(Wsqrt, WA.col(i));
  }

  if (initialize) {
    (*G).noalias() = WA.transpose() * WA;
    sys->AW.noalias() = o->constraint_matrix_.transpose() * W;
    sys->AQc.noalias() = WA.transpose() * WC;
    sys->QwCNorm = (W.cwiseProduct(o->constraint_affine_)).squaredNorm();
    sys->QwCTrace = (W.cwiseProduct(o->constraint_affine_)).sum();
  } else {
    (*G).noalias() += WA.transpose() * WA;
    sys->AW.noalias() += o->constraint_matrix_.transpose() * W;
    sys->AQc.noalias() += WA.transpose() * WC;
    sys->QwCNorm += (W.cwiseProduct(o->constraint_affine_)).squaredNorm();
    sys->QwCTrace += (W.cwiseProduct(o->constraint_affine_)).sum();
  }
}

template <typename T>
void ConservativeResizeHelper(T* constraint_matrix_, int var, int rows) {
  if (!(var < constraint_matrix_->cols())) {
    int cols_new = var + 1 - constraint_matrix_->cols();
    constraint_matrix_->conservativeResize(rows, var + 1);
    constraint_matrix_->rightCols(cols_new).setZero();
  }
}

bool UpdateLinearOperator(SOCConstraint* o, double val, int var, int r, int c,
                          int dim) {
  CONEX_DEMAND(dim == 0, "Complex second-order cone not supported.");
  CONEX_DEMAND(c == 0, "Second-order constraint is not matrix valued.");
  CONEX_DEMAND(r <= o->n_, "Row index out of bounds.");
  CONEX_DEMAND((var >= 0) && (r >= 0), "Indices cannot be negative.");

  ConservativeResizeHelper(&o->constraint_matrix_, var, o->n_ + 1);
  o->constraint_matrix_(r, var) = val;
  return CONEX_SUCCESS;
}

bool UpdateAffineTerm(SOCConstraint* o, double val, int r, int c, int dim) {
  CONEX_DEMAND(dim == 0, "Complex second-order cone not supported.");
  CONEX_DEMAND(c == 0, "Second-order constraint is not matrix valued.");
  CONEX_DEMAND(r <= o->n_, "Row index out of bounds.");
  CONEX_DEMAND(r >= 0, "Indices cannot be negative.");

  ConservativeResizeHelper(&o->constraint_affine_, 0, o->n_ + 1);
  o->constraint_affine_(r) = val;
  return CONEX_SUCCESS;
}

}  // namespace conex
