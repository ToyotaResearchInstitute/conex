namespace conex {
double DivergenceUpperBoundInverse(double divergence_upper_bound,
                                   double gw_norm, double gw_norm_inf,
                                   double gw_trace, int rank);

double DivergenceUpperBound(double k, double gw_norm, double gw_norm_inf,
                            double gw_trace, int n);
}  // namespace conex
