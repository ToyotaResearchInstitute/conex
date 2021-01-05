#include "conex/exponential_map.h"

namespace conex {

namespace {
inline int factorial(int n) {
  int y = 1;
  for (int i = 2; i <= n; i++) {
    y *= i;
  }
  return y;
}
};  // namespace

template <int n>
void DoExponentialMap(const HyperComplexMatrix& xinput, HyperComplexMatrix* y) {
  using T = MatrixAlgebra<n>;

  int squarings = 2;  // Must be power of 2.
  int degree = 2;

  // Initialize to x
  HyperComplexMatrix xpow(n);
  for (int i = 0; i < n; i++) {
    xpow.at(i) = xinput.at(i) * 1.0 / std::pow(2.0, squarings);
  }

  // Initialiaze: y = I + x.
  *y = xpow;
  y->at(0).diagonal().array() += 1;
  for (int i = 2; i <= degree; i++) {
    // Multiply previous term by 1/i x.
    xpow = T::Multiply(xinput, xpow);
    xpow.rescale(1.0 / i * 1.0 / std::pow(2.0, squarings));
    *y = T::Add(*y, xpow);
  }

  for (int i = 0; i < squarings / 2; i++) {
    xpow = T::Multiply(*y, *y);
    *y = T::Multiply(xpow, xpow);
  }
};

void ExponentialMap(const HyperComplexMatrix& arg, HyperComplexMatrix* result) {
  assert(arg.size() > 0);
  assert(arg.size() == result->size());
  int n = arg.size();

  switch (n) {
    case 1:
      DoExponentialMap<1>(arg, result);
      break;
    case 2:
      DoExponentialMap<2>(arg, result);
      break;
    case 4:
      DoExponentialMap<4>(arg, result);
      break;
    case 8:
      DoExponentialMap<8>(arg, result);
      break;
    default:
      throw "Dimension of division algebra must be 1, 2, 4, or 8.";
  }
}

template <typename T>
typename T::Matrix DoGeodesicUpdate(const typename T::Matrix& w,
                                    const typename T::Matrix& s) {
  auto y1 = w;
  auto y2 = T::QuadraticRepresentation(w, s);
  auto y = T::Add(y1, y2);
  for (int i = 1; i < 6; i++) {
    y1 = T::QuadraticRepresentation(w, T::QuadraticRepresentation(s, y1));
    y2 = T::QuadraticRepresentation(w, T::QuadraticRepresentation(s, y2));
    y = T::Add(y, T::Add(T::ScalarMultiply(y1, 1.0 / factorial(2 * i)),
                         T::ScalarMultiply(y2, 1.0 / factorial(2 * i + 1))));

    y1 = T::ScalarMultiply(T::Add(y1, T::ConjugateTranspose(y1)), .5);
    y2 = T::ScalarMultiply(T::Add(y2, T::ConjugateTranspose(y2)), .5);
    y = T::ScalarMultiply(T::Add(y, T::ConjugateTranspose(y)), .5);
  }
  return y;
}

HyperComplexMatrix GeodesicUpdate(const HyperComplexMatrix& x,
                                  const HyperComplexMatrix& s) {
  assert(x.size() > 0);
  assert(x.size() == s.size());
  int n = x.size();

  switch (n) {
    case 1:
      return DoGeodesicUpdate<Real>(x, s);
      break;
    case 2:
      return DoGeodesicUpdate<Complex>(x, s);
      break;
    case 4:
      return DoGeodesicUpdate<Quaternions>(x, s);
      break;
    case 8:
      return DoGeodesicUpdate<Octonions>(x, s);
      break;
    default:
      throw "Dimension of division algebra must be 1, 2, 4, or 8.";
  }
  // Unreachable
  return x;
}

// Evaluates f(w, s) := Q(w^{1/2}) exp(e + Q(w^{1/2} s) using the identity
//
//  f(w, s) = Q(w^{1/2}) (1 + 1/n (e+ Q(w^{1/2} s) )^{n}
//
// for n = 2. Expanding:
//
//  f(w, s) = Q(w^{1/2}) ((1 + 1/2) e + 1/2  Q(w^{1/2} s ) ((1 + 1/2) e + 1/2
//  Q(w^{1/2} s)
//
//           = Q(w^{1/2})  (c e + k Q(w^{1/2}) s)^2,
//
//  for c = 1.5 and k = 2.
//
// Using the fact that:
//
//   [Q(w^{1/2} s)]^2 =  Q(w^{1/2} Q(s) Q(w^{1/2}) e
//
// We conclude that
//
//   f(w, s) = Q(w^{1/2}) (c^2 e + 2 c k Q(w^{1/2}) s +  k^2 Q(w^{1/2}) Q(s) w
//           = (c^2 w + 2  c k Q(w) s +  k^2 Q(w) Q(s) w
template <typename T>
typename T::Matrix DoGeodesicUpdateScaled(const typename T::Matrix& w,
                                          const typename T::Matrix& s) {
  double c = 1.5;
  double k = 1.0 / 2.0;
  return T::MakeHermitian(T::Add(
      T::Add(T::ScalarMultiply(w, c * c),
             T::ScalarMultiply(T::QuadraticRepresentation(w, s), 2 * k * c)),
      T::ScalarMultiply(
          T::QuadraticRepresentation(w, T::QuadraticRepresentation(s, w)),
          k * k)));
}

HyperComplexMatrix GeodesicUpdateScaled(const HyperComplexMatrix& x,
                                        const HyperComplexMatrix& s) {
  assert(x.size() > 0);
  assert(x.size() == s.size());
  int n = x.size();

  switch (n) {
    case 1:
      return DoGeodesicUpdateScaled<Real>(x, s);
      break;
    case 2:
      return DoGeodesicUpdateScaled<Complex>(x, s);
      break;
    case 4:
      return DoGeodesicUpdateScaled<Quaternions>(x, s);
      break;
    case 8:
      return DoGeodesicUpdateScaled<Octonions>(x, s);
      break;
    default:
      throw "Dimension of division algebra must be 1, 2, 4, or 8.";
  }
  // Unreachable
  return x;
}

}  // namespace conex
