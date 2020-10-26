#include "conex/quadratic_cone_constraint.h"
#include "conex/newton_step.h"
using EigenType = DenseMatrix;
using Real = double;


// Implements the spectral decomposition of the Spin Factor algebra.
// See http://rutcor.rutgers.edu/~alizadeh/CLASSES/12fallSDP/Notes/Lecture08/lec08.pdf
// or "Analysis on Symmetric Cones" by Faraut and Koranyi.
namespace {
class SpectralDecompSpinFactorQ {
 public:
  using IdempotentType = DenseMatrix; 
  using EigenType = Eigen::VectorXd; 
  using EssentialVectorType = DenseMatrix; 

  struct PeirceDecompType {
    EigenType X00;
    EigenType X11;
    EigenType X01;
    EigenType Component(int i, int j) {
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
  void Compute(const Eigen::Ref<const EigenType>& x) {
    int n = size - 1;
    assert(x.rows() == n + 1);
    q_ = x.col(0).tail(n);
    norm_of_q_ = std::sqrt(InnerProduct(q_, q_)); 
    if (norm_of_q_ > 0) {
      q_ = q_ / norm_of_q_;
    }
    eigenvalues_(0) = x(0) + norm_of_q_;
    eigenvalues_(1) = x(0) - norm_of_q_;
  }

  DenseMatrix Idempotent(int i) const { return Idempotents().col(i); }

  // Implements equations from page 7 of
  // http://rutcor.rutgers.edu/~alizadeh/CLASSES/12fallSDP/Notes/Lecture08/lec08.pdf
  DenseMatrix Idempotents() const {
    int n = size - 1;
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
    PeirceDecompType peirce_decomp;
    // Compute X00 and X11 by directly computing orthogonal projection
    // onto S00 and S11
    const double c0 = .5 * x(0);
    const double c1 = .5 * InnerProduct(q_, x.tail(size - 1));
    peirce_decomp.X00(0) = c0 + c1;
    peirce_decomp.X00.tail(size - 1) = (c0 + c1) * q_;
    peirce_decomp.X11(0) = c0 - c1;
    peirce_decomp.X11.tail(size - 1) = (c1 - c0) * q_;

    // Compute X01 by using the fact that S01 + S00 + S11 is a direct-sum
    // decomposition
    peirce_decomp.X01 = x - peirce_decomp.X00 - peirce_decomp.X11;
    return peirce_decomp;
  }

  EigenType TransformFromPeirceComponents(const PeirceDecompType& X) const { return X.X00 + X.X11 + X.X01; }

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
  EssentialVectorType EssentialVectorOfDiagonalPeirceComponents(const Eigen::Ref<const EssentialVectorType> z) const {
    const double inner_product = InnerProduct(q_.tail(size - 1), z);
    return q_ * inner_product;
  }

  double InnerProduct(const DenseMatrix& x, const DenseMatrix& y) const {
    return (x.transpose() * Q * y)(0, 0);
  }

  SpectralDecompSpinFactorQ(int n, const DenseMatrix& Qin) : Q(Qin), size(n+1), q_(n, 1)  {}
  DenseMatrix Q;

 private:
  int size;
  Eigen::Matrix<Real, 2, 1> eigenvalues_;
  Eigen::VectorXd q_;
  Real norm_of_q_;
};


double InnerProduct(const DenseMatrix& Q, const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
  int order = x.rows();
  return 2 * (x(0) * y(0) +   x.tail(order-1).transpose() * Q * y.tail(order-1));
}

double SquaredNorm(const DenseMatrix& Q, const DenseMatrix& x) {
  DenseMatrix y = (x.transpose() * Q * x);
  assert(y.rows() == 1);
  assert(y.cols() == 1);
  return y(0, 0); 
}

DenseMatrix QuadraticRepresentation(const DenseMatrix& Q, const Eigen::VectorXd& x,
                                    const Eigen::VectorXd& y) {
    // We use the formula from Example 11.12 of "Formally Real Jordan Algebras
    // and Their Applications to Optimization"  by Alizadeh, which states the quadratic
    // representation of x equals the linear map
    //                          2xx' - (det x) * R
    // where R is the reflection operator R = diag(1, -1, ..., -1) and det x is the determinate
    // of x = (x0, x1), i.e., det x = x0^2 - |x1|^2.
    int order = x.rows();
    double det_x = x(0) * x(0) - SquaredNorm(Q, x.tail(order - 1));
    EigenType z = det_x * y;
    z(0) *= -1;
    return (InnerProduct(Q, x, y)) * x + z;
}

DenseMatrix Sqrt(const DenseMatrix& Q, double x0, const DenseMatrix& x) {
  int n = x.rows();
  DenseMatrix z(n + 1, 1);
  z(0, 0) = x0;
  z.bottomRows(n) = x;
  SpectralDecompSpinFactorQ spec(n, Q);
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

DenseMatrix Exp(const DenseMatrix&Q, double x0, const DenseMatrix& x) {
  int n = x.rows();
  DenseMatrix z(n + 1, 1);
  z(0, 0) = x0;
  z.bottomRows(n) = x;
  SpectralDecompSpinFactorQ spec(n, Q);
  spec.Compute(z);
  auto ev = spec.Eigenvalues();
  DenseMatrix zsqrt = std::exp(ev(0, 0)) * spec.Idempotent(0) + 
                      std::exp(ev(1, 0)) * spec.Idempotent(1);
  return zsqrt;
}


double NormInf(const DenseMatrix&Q, double x0, const DenseMatrix& x) {
  int n = x.rows();
  DenseMatrix z(n + 1, 1);
  z(0, 0) = x0;
  z.bottomRows(n) = x;
  SpectralDecompSpinFactorQ spec(n, Q);
  spec.Compute(z);
  auto ev = spec.Eigenvalues();
  if (std::fabs(ev(0)) > std::fabs(ev(1))) {
    return std::fabs(ev(0));
  } else {
    return std::fabs(ev(1));
  }
}
}

void QuadraticConstraint::ComputeNegativeSlack(double inv_sqrt_mu, const Ref& y, Ref* minus_s) {
  minus_s->noalias() = (constraint_matrix_)*y;
  minus_s->noalias() -= (constraint_affine_) * inv_sqrt_mu;
}

// Combine this with TakeStep
void MinMu(QuadraticConstraint* o,  const Ref& y, MuSelectionParameters* p) {
  auto* workspace = &o->workspace_;
  auto& minus_s = workspace->temp_1;
  auto& Ws = workspace->temp_2;
  o->ComputeNegativeSlack(1, y, &minus_s);

  auto wsqrt = Sqrt(o->Q_, o->workspace_.W0, o->workspace_.W1);
  int n = workspace->n_;
  Ws = QuadraticRepresentation(o->Q_, wsqrt, minus_s);

  SpectralDecompSpinFactorQ spec(n, o->Q_);
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
  p->gw_norm_squared += std::pow(lamda_max, 2) + std::pow(lamda_min, 2);
  p->gw_trace += (lamda_max + lamda_min);
}


void TakeStep(QuadraticConstraint* o, const StepOptions& opt, const Ref& y, StepInfo* info) {
  auto& minus_s = o->workspace_.temp_1;
  o->ComputeNegativeSlack(opt.inv_sqrt_mu, y, &minus_s);

  // e - Q(w^{1/2})(C-A^y)
  auto wsqrt = Sqrt(o->Q_, o->workspace_.W0, o->workspace_.W1);
  int n = wsqrt.rows();
  auto d = QuadraticRepresentation(o->Q_, wsqrt, minus_s);
  d(0, 0) += 1;

  info->norminfd = NormInf(o->Q_, d(0, 0), d.bottomRows(n-1));
  info->normsqrd = d.squaredNorm();

  double scale = info->norminfd * info->norminfd;
  if (scale > 2.0) {
    d = 2 * d / scale;
  } 
  auto expd = Exp(o->Q_, d(0, 0), d.bottomRows(n-1));
  auto wn = QuadraticRepresentation(o->Q_, wsqrt, expd);
  o->workspace_.W0 = wn(0, 0);
  o->workspace_.W1 = wn.bottomRows(n-1);

  if (o->workspace_.W0 < std::sqrt(SquaredNorm(o->Q_, o->workspace_.W1))) {
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



void ConstructSchurComplementSystem(QuadraticConstraint* o, bool initialize, 
                                    SchurComplementSystem* sys) {
  int n = o->workspace_.n_;
  auto Wsqrt = Sqrt(o->Q_, o->workspace_.W0, o->workspace_.W1);
  DenseMatrix W(n+1, 1);
  W(0, 0) = o->workspace_.W0;
  W.bottomRows(n) = o->workspace_.W1;

  DenseMatrix IP = Eigen::MatrixXd::Identity(n+1, n+1);
  IP.bottomRightCorner(n, n) = o->Q_;

  auto G = &sys->G;

  auto A = o->constraint_matrix_;
  auto C = o->constraint_affine_;
  Eigen::MatrixXd WA = o->constraint_matrix_;
  Eigen::MatrixXd WC = QuadraticRepresentation(o->Q_, W, o->constraint_affine_);

  for (int i = 0; i < WA.cols(); i++) {
    WA.col(i) = QuadraticRepresentation(o->Q_, W, WA.col(i));
  }
  
  if (initialize) {
    (*G).noalias() = A.transpose() *IP *WA;
    sys->AW.noalias()  = A.transpose() *IP * W;
    sys->AQc.noalias() = A.transpose() *IP * WC; 
    sys->QwCNorm = (W.cwiseProduct(o->constraint_affine_)).squaredNorm();
    sys->QwCTrace = (W.cwiseProduct(o->constraint_affine_)).sum();
  } else {
    (*G).noalias() += A.transpose() *IP *WA;
    sys->AW.noalias()  += A.transpose() *IP* W;
    sys->AQc.noalias() += A.transpose() *IP* WC; 

    // TODO Are these correct?
    sys->QwCNorm += (W.cwiseProduct(o->constraint_affine_)).squaredNorm();
    sys->QwCTrace += (W.cwiseProduct(o->constraint_affine_)).sum();
  }
  // DUMP(*G);
  // auto T = *G;
  // auto T12 = G->bottomLeftCorner(n, 1);
  // auto T22 = G->bottomRightCorner(n, n);
  // DUMP(T22 - T12* T12.transpose() * 1.0/T(0, 0));

  // DUMP(o->Q_);
}

 // A Q A
