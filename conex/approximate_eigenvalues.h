#include <Eigen/Dense>

namespace conex {


// Approximate eigenvalues of symmetric S.
Eigen::VectorXd ApproximateEigenvalues(const Eigen::MatrixXd& S,
                                       const Eigen::MatrixXd& r0,
                                       int num_iterations);

// Approximate eigenvalues of SW where S is symmetric and W is PSD.
Eigen::VectorXd ApproximateEigenvalues(const Eigen::MatrixXd& S,
                                       const Eigen::MatrixXd& W,
                                       const Eigen::MatrixXd& r,
                                       int num_iterations, bool compressed);

} // namespace conex
