#include "conex/exponential_map.h"


//  I + X + X^2

template<int n>
void DoExponentialMap(const HyperComplexMatrix& xinput, HyperComplexMatrix* y) {
  using T = MatrixAlgebra<n>;

  int squarings = 2*4;
  // Initialize to x
  HyperComplexMatrix xpow(n);
  for (int i = 0; i < n; i++) {
    xpow.at(i) = xinput.at(i) * 1.0/std::pow(2.0, squarings);
  }

  // y = I + x. 
  *y = xpow;
  y->at(0).diagonal().array() += 1;
  for (int i = 2; i < 5; i++) {
    // Multiply previous term by 1/i x.
    xpow = T::Multiply(xinput, xpow);
    xpow.rescale(1.0/i  *  1.0/std::pow(2.0, squarings)  );
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
  int dimension = arg.at(0).size();
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
      bool valid_arguments = false;
      assert(valid_arguments);
  }
}
