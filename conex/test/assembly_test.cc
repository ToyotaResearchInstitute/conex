#include <any>

#include "conex/clique_ordering.h"
#include "conex/constraint_manager.h"
#include "conex/debug_macros.h"
#include "conex/kkt_system_assembler.h"
#include "conex/kkt_solver.h"

#include "gtest/gtest.h"

#include "conex/block_triangular_operations.h"
#include "conex/equality_constraint.h"
#include "conex/kkt_assembler.h"
#include "conex/supernodal_solver.h"

#include <chrono>

namespace conex {


class Container {
 public:
  template<typename T>
  Container(const T& x) : obj(x), 
                         kkt(std::any_cast<T>(&obj)) {}
  std::any obj;
  KKT_SystemAssembler kkt;
};


std::vector<KKT_SystemAssembler*> GetPointers(std::list<Container>& r) {
  std::vector<KKT_SystemAssembler*> y;
  for (auto& ri: r) {
    y.push_back(&ri.kkt);
    //y.push_back(&ri.kkt);
  }
  return y;
}


using Cliques = std::vector<std::vector<int>>;
using Eigen::MatrixXd;
using std::vector;

int GetMax(const std::vector<Clique>& cliques) {
  int max = cliques.at(0).at(0);
  for (const auto& c : cliques) {
    for (const auto ci : c) {
      if (ci > max) {
        max = ci;
      }
    }
  }
  return max;
}

void Assign(Eigen::MatrixXd& T, vector<int>& permutation, vector<int>& rows,
            vector<int>& cols, Eigen::Map<MatrixXd>* matrix) {
  int i = 0;
  for (auto& ui : rows) {
    int j = 0;
    for (auto& vj : cols) {
      (*matrix)(i, j) = T(permutation.at(ui), permutation.at(vj));
      j++;
    }
    i++;
  }
}

//vector<int> Relabel(const vector<int>& x, const vector<int>& labels) {
//  vector<int> y;
//  for (auto& xi : x) {
//    y.push_back(labels.at(xi));
//  }
//  return y;
//}
//
//
//std::vector<int> ReplaceWithPosition(const std::vector<int>& a,
//                                     const std::vector<int>& b) {
//  std::vector<int> y;
//  for (auto ai : a) {
//    auto p = std::find(b.begin(), b.end(), ai);
//    if (p != b.end()) {
//      y.push_back(std::distance(b.begin(), p));
//    } else {
//      DUMP(a);
//      DUMP(b);
//      bool found = 0;
//      assert(found);
//    }
//  }
//  return y;
//}
//
//std::vector<int> ConcatFirstN(const std::vector<int>& a, int N,
//                              const std::vector<int>& b) {
//  std::vector<int> y;
//  for (int i = 0; i < N; i++) {
//    y.push_back(a.at(i));
//  }
//  for (size_t j = 0; j < b.size(); j++) {
//    y.push_back(b.at(j));
//  }
//  return y;
//}

void BuildLQRProblem(int N, ConstraintManager<Container>* prg) {
  auto& prog = *prg;
  Eigen::MatrixXd Qi = Eigen::MatrixXd::Identity(3, 3) * 2;

  MatrixXd A0(2, 3);
  MatrixXd Ai(2, 5);
  MatrixXd bi(2, 1); bi << 1, 2;

  // clang-format off
  A0 << 1, 1, 0,
        1, 0, 1;

  Ai << 1, 1, 1, 1, 0,
        1, 1, 1, 0, 1;
  // clang-format on

  int max_var = (N + 1) * (2 + 1);

  prog.SetNumberOfVariables(max_var);
  prog.AddConstraint(EqualityConstraints{A0, bi}, vector{0, 1, 2});

  int o = 0;
  for (int i = 0; i < N; i++) {
    vector vars{1 + o, 2 + o, 3 + o, 4 + o, 5 + o};
    o += 3;
    prog.AddConstraint(EqualityConstraints{Ai * (i + 2), bi * (i+2)}, vars);
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

#if 0
TEST(LDLT, Benchmark) {
  using Eigen::MatrixXd;
  constexpr int m = 6;
  constexpr int n = 9;
  Eigen::MatrixXd A(m, n);
  // 0  1  2  3  4  5  6  7  8
  A << 1, 1, 0, 0, 0, 0, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 2, 2, 2, 2, 0, 0, 0, 0,
       0, 2, 2, 2, 0, 2, 0, 0, 0,
       0, 0, 0, 0, 3, 3, 3, 3, 0,
       0, 0, 0, 0, 3, 3, 3, 0, 3;

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n, n) * 4;
  Eigen::MatrixXd Qi = Eigen::MatrixXd::Identity(3, 3) * 2;


  int N = 2;
  int max_var = (N + 1) * (2 + 1);

  ConstraintManager prog;
  BuildLQRProblem(N, &prog);

  auto cliques = prog.cliques;
  auto& eqs = prog.eqs;

  Sort(&cliques);
  vector<int> order;
  vector<vector<int>> separators;
  vector<vector<int>> supernodes;

  MatrixData data;
  TIME(data = GetData(cliques););
  SparseTriangularMatrix mat(data);

  data.supernodes_original_labels.resize(cliques.size());
  data.separators_original_labels.resize(cliques.size());
  for (int e = 0; e < static_cast<int>(cliques.size()); e++) {
    data.supernodes_original_labels.at(e) = 
        Relabel(data.supernodes.at(e), data.permutation_inverse);
    data.separators_original_labels.at(e) = 
        Relabel(data.separators.at(e), data.permutation_inverse);
  }

  for (int e = 0; e < static_cast<int>(cliques.size()); e++) {
    int j = data.clique_order.at(e);
    auto labels = ConcatFirstN(prog.cliques.at(j),
                               prog.cliques.at(j).size() - prog.dual_vars.at(j).size(),
                               prog.dual_vars.at(j));

    data.supernodes_original_labels.at(e) =
        ReplaceWithPosition(data.supernodes_original_labels.at(e), labels);

    data.separators_original_labels.at(e) =
        ReplaceWithPosition(data.separators_original_labels.at(e), labels);

  }

  Bind(data, mat.workspace_, eqs);

  // Set to random numbers to test initialization.
  for (int e = 0; e < static_cast<int>(cliques.size()); e++) {
    mat.workspace_.off_diagonal.at(e).setConstant(-3);
    mat.workspace_.diagonal.at(e).setConstant(-2);
  }

  for (int e = static_cast<int>(cliques.size()) - 1; e >=0; e--) {
    int i = data.clique_order.at(e);
    eqs.at(i).UpdateBlocks();
  }
  
  MatrixXd T = TriangularMatrixOperations::ToDense(mat).selfadjointView<Eigen::Lower>();




  using Eigen::VectorXd;
  VectorXd b(T.rows()); b.setConstant(1);
  VectorXd yref;

  using LLT = Eigen::LDLT<Eigen::Ref<MatrixXd>, Eigen::Lower>;
  TIME(LLT ldlt(T); ldlt.compute(T); yref = ldlt.solve(b););

  

  VectorXd y = b;
  TIME(std::vector<Eigen::LDLT<Eigen::Ref<MatrixXd>>> factorization;
  BlockTriangularOperations::BlockLDLTInPlace(&mat.workspace_, &factorization);
  BlockTriangularOperations::SolveInPlaceLDLT(mat.workspace_, factorization, &y););

  DUMP(y);
  DUMP(yref);
  EXPECT_NEAR( (y-yref).norm(), 0, 1e-9);

}
#endif

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

}  // namespace conex
