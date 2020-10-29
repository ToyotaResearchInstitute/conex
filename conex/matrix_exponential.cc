#include "matrix_exponential.h"
#include "debug_macros.h"
#include <cmath>
#include <complex>
using RealScalar = double;
/** \brief Compute the (3,3)-Pad&eacute; approximant to the exponential.
 *
 *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
 *  approximant of \f$ \exp(A) \f$ around \f$ A = 0 \f$.
 */
template <typename MatA, typename MatU, typename MatV>
void matrix_exp_pade3(const MatA& A, MatU& U, MatV& V)
{
  DUMP("3");
  typedef typename MatA::PlainObject MatrixType;
  const RealScalar b[] = {120.L, 60.L, 12.L, 1.L};
  const MatrixType A2 = A * A;
  const MatrixType tmp = b[3] * A2 + b[1] * MatrixType::Identity(A.rows(), A.cols());
  U.noalias() = A * tmp;
  V = b[2] * A2 + b[0] * MatrixType::Identity(A.rows(), A.cols());
}

/** \brief Compute the (5,5)-Pad&eacute; approximant to the exponential.
 *
 *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
 *  approximant of \f$ \exp(A) \f$ around \f$ A = 0 \f$.
 */
template <typename MatA, typename MatU, typename MatV>
void matrix_exp_pade5(const MatA& A, MatU& U, MatV& V)
{
  DUMP("5");
  typedef typename MatA::PlainObject MatrixType;
  const RealScalar b[] = {30240.L, 15120.L, 3360.L, 420.L, 30.L, 1.L};
  const MatrixType A2 = A * A;
  const MatrixType A4 = A2 * A2;
  const MatrixType tmp = b[5] * A4 + b[3] * A2 + b[1] * MatrixType::Identity(A.rows(), A.cols());
  U.noalias() = A * tmp;
  V = b[4] * A4 + b[2] * A2 + b[0] * MatrixType::Identity(A.rows(), A.cols());
}

/** \brief Compute the (7,7)-Pad&eacute; approximant to the exponential.
 *
 *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
 *  approximant of \f$ \exp(A) \f$ around \f$ A = 0 \f$.
 */
template <typename MatA, typename MatU, typename MatV>
void matrix_exp_pade7(const MatA& A, MatU& U, MatV& V)
{
  DUMP("7");
  typedef typename MatA::PlainObject MatrixType;
  const RealScalar b[] = {17297280.L, 8648640.L, 1995840.L, 277200.L, 25200.L, 1512.L, 56.L, 1.L};
  const MatrixType A2 = A * A;
  const MatrixType A4 = A2 * A2;
  const MatrixType A6 = A4 * A2;
  const MatrixType tmp = b[7] * A6 + b[5] * A4 + b[3] * A2 
    + b[1] * MatrixType::Identity(A.rows(), A.cols());
  U.noalias() = A * tmp;
  V = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * MatrixType::Identity(A.rows(), A.cols());

}


/** \brief Compute the (9,9)-Pad&eacute; approximant to the exponential.
 *
 *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
 *  approximant of \f$ \exp(A) \f$ around \f$ A = 0 \f$.
 */
template <typename MatA, typename MatU, typename MatV>
void matrix_exp_pade9(const MatA& A, MatU& U, MatV& V)
{
  typedef typename MatA::PlainObject MatrixType;
  const RealScalar b[] = {17643225600.L, 8821612800.L, 2075673600.L, 302702400.L, 30270240.L,
                          2162160.L, 110880.L, 3960.L, 90.L, 1.L};
  //MatrixType tmp = b[1] * MatrixType::Identity(A.rows(), A.cols()); 
  //V = b[0] * MatrixType::Identity(A.rows(), A.cols());

  //const MatrixType A2 = A * A;
  //tmp += b[3] * A2;
  //V += b[2] * A2;

  //const MatrixType A4 = A2 * A2;
  //tmp += b[5] * A4;
  //V += b[4] * A4;

  //const MatrixType A6 = A4 * A2;
  //tmp += b[7] * A6;
  //V += b[6] * A6;

  //const MatrixType A8 = A6 * A2;
  //tmp += b[9] * A8;
  //V += b[8] * A8;

  ////const MatrixType tmp = b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * MatrixType::Identity(A.rows(), A.cols());


  //U.noalias() = A * tmp;
  // V = b[8] * A8 + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * MatrixType::Identity(A.rows(), A.cols());
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //

#if 0
  MatrixType Apow1;
  MatrixType Apow2;
  V = b[0] * MatrixType::Identity(A.rows(), A.cols());
  U = b[1] * A; 

  Apow1 = A * A;
  V += b[2] * Apow1;
  Apow2 = A * Apow1;
  U += b[3] * Apow2;

  Apow1 = A * Apow2;
  V += b[4] * Apow1;
  Apow2 = A * Apow1;
  U += b[5] * Apow2;

  Apow1 = A * Apow2;
  V += b[6] * Apow1;
  Apow2 = A * Apow1;
  U += b[7] * Apow2;

  Apow1 = A * Apow2;
  V += b[8] * Apow1;
  Apow2 = A * Apow1;
  U += b[9] * Apow2;
#else
  MatrixType Asqr;
  MatrixType Apow;
  MatrixType Apow1;
  V = b[0] * MatrixType::Identity(A.rows(), A.cols());
  U = b[1] * MatrixType::Identity(A.rows(), A.cols());

  Asqr = A * A;
  V += b[2] * Asqr;
  U += b[3] * Asqr;

  Apow = Asqr * Asqr;
  V += b[4] * Apow;
  U += b[5] * Apow;

  Apow1 = Asqr * Apow;
  V += b[6] * Apow1;
  U += b[7] * Apow1;

  Apow = Asqr * Apow1;
  V += b[8] * Apow;
  U += b[9] * Apow;
  U = A * U;
#endif
}

/** \brief Compute the (13,13)-Pad&eacute; approximant to the exponential.
 *
 *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
 *  approximant of \f$ \exp(A) \f$ around \f$ A = 0 \f$.
 */
template <typename MatA, typename MatU, typename MatV>
void matrix_exp_pade13(const MatA& A, MatU& U, MatV& V)
{
  DUMP("13");
  typedef typename MatA::PlainObject MatrixType;
  const RealScalar b[] = {64764752532480000.L, 32382376266240000.L, 7771770303897600.L,
                          1187353796428800.L, 129060195264000.L, 10559470521600.L, 670442572800.L,
                          33522128640.L, 1323241920.L, 40840800.L, 960960.L, 16380.L, 182.L, 1.L};
  const MatrixType A2 = A * A;
  const MatrixType A4 = A2 * A2;
  const MatrixType A6 = A4 * A2;
  V = b[13] * A6 + b[11] * A4 + b[9] * A2; // used for temporary storage
  MatrixType tmp = A6 * V;
  tmp += b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * MatrixType::Identity(A.rows(), A.cols());
  U.noalias() = A * tmp;
  tmp = b[12] * A6 + b[10] * A4 + b[8] * A2;
  V.noalias() = A6 * tmp;
  V += b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * MatrixType::Identity(A.rows(), A.cols());
}

template <typename RealScalar>
struct MatrixExponentialScalingOp
{
  /** \brief Constructor.
   *
   * \param[in] squarings  The integer \f$ s \f$ in this document.
   */
  MatrixExponentialScalingOp(int squarings) : m_squarings(squarings) { }


  /** \brief Scale a matrix coefficient.
   *
   * \param[in,out] x  The scalar to be scaled, becoming \f$ 2^{-s} x \f$.
   */
  inline const RealScalar operator() (const RealScalar& x) const
  {
    using std::ldexp;
    return ldexp(x, -m_squarings);
  }

  typedef std::complex<RealScalar> ComplexScalar;

  /** \brief Scale a matrix coefficient.
   *
   * \param[in,out] x  The scalar to be scaled, becoming \f$ 2^{-s} x \f$.
   */
  inline const ComplexScalar operator() (const ComplexScalar& x) const
  {
    using std::ldexp;
    return ComplexScalar(ldexp(x.real(), -m_squarings), ldexp(x.imag(), -m_squarings));
  }

  private:
    int m_squarings;
};

template<typename ArgType, typename MatrixType>
void run(const ArgType& arg, MatrixType& U, MatrixType& V, int& squarings)
{
  using std::frexp;
  using std::pow;
  const RealScalar l1norm = arg.cwiseAbs().colwise().sum().maxCoeff();
  squarings = 0;
  if (l1norm < 1.495585217958292e-002) {
    matrix_exp_pade3(arg, U, V);
  } else if (l1norm < 2.539398330063230e-001) {
    matrix_exp_pade5(arg, U, V);
  } else if (l1norm < 9.504178996162932e-001) {
    matrix_exp_pade7(arg, U, V);
  } else if (l1norm < 2.097847961257068e+000) {
    matrix_exp_pade9(arg, U, V);
  } else {
    const RealScalar maxnorm = 5.371920351148152;
    frexp(l1norm / maxnorm, &squarings);
    if (squarings < 0) squarings = 0;
    MatrixType A = arg.unaryExpr(MatrixExponentialScalingOp<RealScalar>(squarings));
    matrix_exp_pade13(A, U, V);
  }
}

void matrix_exp_compute(const Ref& arg, Ref &result) // natively supported scalar type
{
  using MatrixType = Ref;
  MatrixType U, V;
  int squarings;
  run(arg, U, V, squarings); // Pade approximant is (U+V) / (-U+V)
  MatrixType numer = U + V;
  MatrixType denom = -U + V;
  result = denom.partialPivLu().solve(numer);
  for (int i=0; i<squarings; i++)
    result *= result;   // undo scaling by repeated squaring
}
