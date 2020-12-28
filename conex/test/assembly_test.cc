#include <any>

#include "conex/clique_ordering.h"
#include "conex/constraint_manager.h"
#include "conex/debug_macros.h"
#include "conex/kkt_solver.h"
#include "conex/kkt_system_assembler.h"

#include "gtest/gtest.h"

#include "conex/block_triangular_operations.h"
#include "conex/equality_constraint.h"
#include "conex/kkt_assembler.h"
#include "conex/supernodal_solver.h"

#include <chrono>

namespace conex {

class Container {
 public:
  template <typename T>
  Container(const T& x) : obj(x), kkt(std::any_cast<T>(&obj)) {}
  std::any obj;
  KKT_SystemAssembler kkt;
};

std::vector<KKT_SystemAssembler*> GetPointers(std::list<Container>& r) {
  std::vector<KKT_SystemAssembler*> y;
  for (auto& ri : r) {
    y.push_back(&ri.kkt);
    // y.push_back(&ri.kkt);
  }
  return y;
}

using Eigen::MatrixXd;
using std::vector;

void BuildLQRProblem(int N, ConstraintManager<Container>* prg) {
  auto& prog = *prg;
  Eigen::MatrixXd Qi = Eigen::MatrixXd::Identity(3, 3) * 2;

  MatrixXd A0(2, 3);
  MatrixXd Ai(2, 5);
  MatrixXd bi(2, 1);
  bi << 1, 2;

  // clang-format off
  A0 << 1, 1, 0,
        1, 0, 1;

  Ai << 1, 1, 1, 1, 0,
        1, 1, 1, 0, 1;
  // clang-format on

  int max_var = (N + 1) * (2 + 1);

  prog.SetNumberOfVariables(max_var);
  prog.AddEqualityConstraint(EqualityConstraints{A0, bi}, vector{0, 1, 2});

  int o = 0;
  for (int i = 0; i < N; i++) {
    vector vars{1 + o, 2 + o, 3 + o, 4 + o, 5 + o};
    o += 3;
    prog.AddEqualityConstraint(EqualityConstraints{Ai * (i + 2), bi * (i + 2)},
                               vars);
  }

  prog.AddConstraint(LinearKKTAssemblerStatic{Qi}, vector{0, 1, 2});

  o = 3;
  for (int i = 0; i < N; i++) {
    vector vars{o, 1 + o, 2 + o};
    o += 3;
    prog.AddConstraint(LinearKKTAssemblerStatic{Qi}, vars);
  }
}

TEST(LDLT, TestAssembly) {
  using Eigen::MatrixXd;
  constexpr int m = 6;
  constexpr int n = 9;
  Eigen::MatrixXd A(m, n);
  // clang-format off
  //   0  1  2  3  4  5  6  7  8
  A << 1, 1, 0, 0, 0, 0, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 2, 2, 2, 2, 0, 0, 0, 0,
       0, 2, 2, 2, 0, 2, 0, 0, 0,
       0, 0, 0, 0, 3, 3, 3, 3, 0,
       0, 0, 0, 0, 3, 3, 3, 0, 3;

  Eigen::VectorXd b(n+m);
  b.setZero();
  b.bottomRows(m) << 1, 2, 
                     2, 4,
                     3, 6;
  // clang-format on

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n, n) * 4;
  Eigen::MatrixXd Qi = Eigen::MatrixXd::Identity(3, 3) * 2;

  int N = 2;

  ConstraintManager<Container> prog;
  BuildLQRProblem(N, &prog);

  EXPECT_EQ(n + m, prog.SizeOfKKTSystem());
  MatrixXd T(n + m, n + m);
  T.setZero();
  T.block(0, 0, n, n) = Q;
  T.block(0, n, n, m) = A.transpose();
  T.block(n, 0, m, n) = A;

  using Eigen::VectorXd;
  Eigen::LDLT<MatrixXd> ldlt;
  VectorXd yref;
  ldlt.compute(T);

  Solver solver(prog.cliques, prog.dual_vars);
  solver.Bind(GetPointers(prog.eqs));
  Eigen::VectorXd AW(n + m);
  Eigen::VectorXd AQc(n + m);
  solver.Assemble(&AW, &AQc);

  MatrixXd error = (solver.KKTMatrix() - T);
  EXPECT_EQ(error.norm(), 0);

  error = (AQc - b);
  EXPECT_EQ(error.norm(), 0);

  solver.Factor();
  for (int i = 0; i < 3; i++) {
    yref = ldlt.solve(b);
    VectorXd y = solver.Solve(b);
    EXPECT_NEAR((y - yref).norm(), 0, 1e-9);
    b = y;
  }
}

TEST(LDLT, Benchmark2) {
  using Eigen::MatrixXd;

  int N = 40;

  ConstraintManager<Container> prog;
  BuildLQRProblem(N, &prog);

  Solver solver(prog.cliques, prog.dual_vars);
  solver.Bind(GetPointers(prog.eqs));
  Eigen::VectorXd AW(prog.SizeOfKKTSystem());
  Eigen::VectorXd AQc(prog.SizeOfKKTSystem());
  solver.Assemble(&AW, &AQc);
  Eigen::MatrixXd T = solver.KKTMatrix().selfadjointView<Eigen::Lower>();

  Eigen::VectorXd b(prog.SizeOfKKTSystem());
  b.setConstant(1);
  solver.Factor();
  for (int i = 0; i < 3; i++) {
    Eigen::VectorXd y = solver.Solve(b);
    EXPECT_NEAR((T * y - b).norm(), 0, 1e-9);
  }
}

#if 0
TEST(bad, bad) {
  MatrixXd Qi = MatrixXd::Random(3, 3);
  Qi.setConstant(1);
  MatrixXd Q2 = MatrixXd::Identity(2, 2);
  Q2(0, 0) = 1000;

  ConstraintManager<Container> prog;
  prog.SetNumberOfVariables(4);
  // prog.AddConstraint(LinearKKTAssemblerStatic{Qi}, vector{2, 0, 1});
  // prog.AddConstraint(LinearKKTAssemblerStatic{Qi}, vector{3, 2, 1});
  // prog.AddConstraint(LinearKKTAssemblerStatic{Q2}, vector{2, 1});

  prog.AddConstraint(LinearKKTAssemblerStatic{MatrixXd::Identity(3, 3)},
                     vector{0, 1, 2});
  prog.AddConstraint(LinearKKTAssemblerStatic{Qi}, vector{3, 2, 1});
  // prog.AddConstraint(LinearKKTAssemblerStatic{Qi}, vector{1, 2, 3});

  // prog.AddConstraint(LinearKKTAssemblerStatic{Q2}, vector{1, 2});

  DUMP(prog.cliques);
  Solver solver(prog.cliques, prog.dual_vars);
  solver.Bind(GetPointers(prog.eqs));
  Eigen::VectorXd AW(prog.SizeOfKKTSystem());
  Eigen::VectorXd AQc(prog.SizeOfKKTSystem());
  solver.Assemble(&AW, &AQc);
  DUMP(solver.KKTMatrix());
}
#endif
}  // namespace conex
