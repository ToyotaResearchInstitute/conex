#include <Eigen/Dense>
using Ref = Eigen::MatrixXd;
void matrix_exp_compute(const Ref& arg, Ref &result);


// Approximate: 
//   A = Q J Q^T
//   exp A = Q exp J Q
// 
// See Method 20: https://www.math.purdue.edu/~yipn/543/matrixExp19-II.pdf 
// 
// Reference: https://core.ac.uk/download/pdf/197542384.pdf: krylov methods
// for approximating exp A

