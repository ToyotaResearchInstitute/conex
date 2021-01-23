#include "conex/newton_step.h"
namespace conex {

// Return -1 if divergence_upper_bound is not in range of
// DivergenceUpperBound.
double DivergenceUpperBoundInverse(double divergence_upper_bound,
                                   MuSelectionParameters& p);

double DivergenceUpperBound(double k, MuSelectionParameters& p);
}  // namespace conex
