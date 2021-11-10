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
  Container(const T& x, int num_vars) : obj(x), kkt(std::any_cast<T>(&obj)) {
    kkt.SetNumberOfVariables(num_vars);
    memory.resize(SizeOf(*kkt.GetWorkspace()));
    Initialize(kkt.GetWorkspace(), memory.data());
  }

  using T = Container;
  Container(const Container&) = delete;
  Container(Container&&) = delete;
  Container& operator=(const Container&) = delete;
  Container& operator=(Container&&) = delete;

  std::any obj;
  Eigen::VectorXd memory;
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
using Eigen::VectorXd;
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

GTEST_TEST(LDLT, TestAssembly) {
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

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n, n) * 2;
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

  SupernodalKKTSolver solver(prog.cliques, prog.dual_vars);
  solver.Bind(GetPointers(prog.eqs));
  Eigen::VectorXd AW(n + m);
  Eigen::VectorXd AQc(n + m);
  double c_inner_product_w;
  solver.Assemble(&AW, &AQc, &c_inner_product_w);

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

GTEST_TEST(LDLT, Benchmark2) {
  using Eigen::MatrixXd;

  int N = 40;

  ConstraintManager<Container> prog;
  BuildLQRProblem(N, &prog);

  SupernodalKKTSolver solver(prog.cliques, prog.dual_vars);
  solver.Bind(GetPointers(prog.eqs));
  Eigen::VectorXd AW(prog.SizeOfKKTSystem());
  Eigen::VectorXd AQc(prog.SizeOfKKTSystem());
  double c_inner_product_w;
  solver.Assemble(&AW, &AQc, &c_inner_product_w);
  Eigen::MatrixXd T = solver.KKTMatrix().selfadjointView<Eigen::Lower>();

  Eigen::VectorXd b(prog.SizeOfKKTSystem());
  b.setConstant(1);
  solver.Factor();
  for (int i = 0; i < 3; i++) {
    Eigen::VectorXd y = solver.Solve(b);
    EXPECT_NEAR((T * y - b).norm(), 0, 1e-9);
  }
}

GTEST_TEST(Assemble, VariablesSpecifiedOutOfOrder) {
  MatrixXd Q = MatrixXd::Identity(3, 3);

  ConstraintManager<Container> prog;
  prog.SetNumberOfVariables(4);
  Q << 1, 0, 0, 0, 0, 0, 0, 0, 3;

  prog.AddConstraint(LinearKKTAssemblerStatic{Q}, vector{1, 0, 3});
  Q << 1, 0, 0, 0, 0, 0, 0, 0, 2;
  prog.AddConstraint(LinearKKTAssemblerStatic{Q}, vector{1, 0, 2});

  SupernodalKKTSolver solver(prog.cliques, prog.dual_vars);
  solver.Bind(GetPointers(prog.eqs));
  Eigen::VectorXd AW(prog.SizeOfKKTSystem());
  Eigen::VectorXd AQc(prog.SizeOfKKTSystem());
  double c_inner_product_w;
  solver.Assemble(&AW, &AQc, &c_inner_product_w);
  auto M = solver.KKTMatrix();
  Eigen::VectorXd expected(4);
  expected << 0, 2, 2, 3;
  EXPECT_EQ((expected - M.diagonal()).norm(), 0);
  MatrixXd Mref = expected.asDiagonal();
  EXPECT_EQ((Mref - M).norm(), 0);
}
}  // namespace conex
