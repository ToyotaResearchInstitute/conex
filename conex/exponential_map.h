#include "conex/vect_jordan_matrix_algebra.h"

namespace conex {

void ExponentialMap(const HyperComplexMatrix& arg, HyperComplexMatrix* result);

// Returns an approximation of Q(w^{1/2}) exp Q(w^{1/2}) s, where Q
// denotes the quadratic representation.
Octonions::Matrix GeodesicUpdate(const Octonions::Matrix& w,
                                 const Octonions::Matrix& s);
HyperComplexMatrix GeodesicUpdateScaled(const HyperComplexMatrix& w,
                                        const HyperComplexMatrix& s);

}  // namespace conex
