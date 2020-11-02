#include "gtest/gtest.h"
#include "conex/cholesky_decomposition.h"
#include "conex/elimination_ordering.h" 
using Matrix = Eigen::MatrixXd;
Matrix Ones(int n) {
  Matrix x(n, n);
  x.setConstant(1);
  return x;
}
int GetMax(std::vector<Clique>& cliques) {
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

std::vector<int> PEC(const MatrixXd& M) {
  DUMP(M);
  auto Pmat = conex::EliminationOrdering(M);
  std::vector<int> order(M.rows());
  for (int i = 0; i < M.rows(); i++) {
    order.at(Pmat.indices()(i)) = i;
  }
  return order;
}

std::vector<int> ChordalMFS(int N, const std::vector<Clique>& cliques) {
  DUMP(cliques);
  Eigen::MatrixXd M(N, N);
  for (unsigned i = 0; i < cliques.size(); i++) {
    for (unsigned j = 0; j < cliques.size(); j++) {
      std::vector<int> intersection;
      IntersectionOfSorted(cliques.at(i), cliques.at(j), &intersection);
      if (intersection.size() > 0) {
        M(i, j) = 1;  
      }
    }
  }
  return PEC(M);
}

#if 0
int main2() {
  int N = 8;
  std::vector<Matrix> Mcliques;
  std::vector<Clique> cliques;


  cliques.push_back({0, 1, 2, 3, 4, 5});
  Mcliques.push_back(50*Ones(cliques.back().size()));

  cliques.push_back({3, 4, 5, 6, 7});
  Mcliques.push_back(20*Ones(cliques.back().size()));

  //cliques.push_back({3, 5, 8, 11});
  //Mcliques.push_back(3*Ones(cliques.back().size()));
////
  //cliques.push_back({0, 2, 4, 6, 8});
  //Mcliques.push_back(Ones(cliques.back().size()));
  //Mcliques.back().diagonal().setConstant(4);
  //Mcliques.back()(0, 4) = 3;
  //Mcliques.back()(4, 0) = 3;

  Eigen::LLT<Matrix> llt(Mcliques.back());
  DUMP(Matrix(llt.matrixL()));

  // cliques.push_back({7, 9, 10, 11, 2});
  // Mcliques.push_back(40*Ones(cliques.back().size()));
  auto Mref = GetMatrix(N, cliques, Mcliques);

  DUMP(GetMatrix(N, cliques, Mcliques));
  // std::vector<float> d = Degree(N, cliques);
  // std::vector<int> P(N);
  // for (int i = 0; i < N; i++) {
  //   P.at(i) = i;
  // }
  // DUMP(d);
  // // want cliq, clique, clique, common.
  // std::sort(P.data(),
  //           P.data() + P.size(),
  //           [&d](const int i, const int j) -> bool {
  //             return d.at(i) < d.at(j);
  //           });
  // DUMP(P);
  // std::vector<int> order(N);
  // for (int i = 0; i < N; i++) {
  //   order.at(P.at(i)) = i;
  // }
  // Eigen::PermutationMatrix<-1> Pmat(N);
  // Pmat.indices() = Eigen::Map<Eigen::VectorXi>(P.data(), N);


  auto Pmat = conex::EliminationOrdering(GetMatrix(N, cliques, Mcliques));
  std::vector<int> order(N);
  for (int i = 0; i < N; i++) {
    order.at(Pmat.indices()(i)) = i;
  }



  for (auto& c : cliques) {
    for (auto& ci : c) {
      ci = order.at(ci);
    }
  }

//  DUMP(Eigen::MatrixXf(Pmat.transpose()* Eigen::Map<Eigen::VectorXf>(d.data(), N)));
//  DUMP(Pmat.transpose() * Mref * Pmat);
  DUMP(GetMatrix(N, cliques, Mcliques));
  Matrix Gper = GetMatrix(N, cliques, Mcliques);
  Eigen::LLT<Matrix> lltper(Gper);
  DUMP(Matrix(lltper.matrixL()));

  DUMP(conex::IsPerfectlyOrdered(Pmat.transpose() * Mref * Pmat));

  return 0;
}

#endif



TEST(Cholesky, TestArrow) {
  //  o - o - o
  //
  //  (1, 2, 3*, -6, -7)
  //  (3, 4, 5*, -6, -7)
  //  (5, 6, 7)
  int n = 7;
  Matrix A(n, n);

  A << 2, 1, 1, 0, 0, 1, 1,
       1, 3, 1, .0, 0, 1, 1,
       1, 1, 5, 3, 3, 1, 1,
       0, .0, 3, 5, 3, 1, 1,
       0, 0, 3, 3, 5, 1, 1,
       1, 1, 1, 1, 1, 4, 1,
       1, 1, 1, 1, 1, 1, 8;

  std::vector<int> cliquestart{0, 2, 4};
  std::vector<int> rows{3, 3, 3};
  std::vector<int> cols{3, 3, 3};
  vector<vector<int>> root{ {5, 6}, {}, {} };


  Matrix temp = A;
  Matrix R(n, n);
  R.setZero();

  SparseCholeskyDecomposition(A, cliquestart, rows, root, &R);
  EXPECT_TRUE((R*R.transpose() - A).norm() < 1e-8);
}

TEST(Cholesky, TestDecomp) {
  int n = 7;
  Matrix A(n, n);
  A << 2, 1, 1, 0, 0, 0, 0,
       1, 3, 1, .1, 0, 0, 0,
       1, 1, 5, 3, 3, 0, 0,
       0, .1, 3, 5, 3, 0, 0,
       0, 0, 3, 3, 5, 1, 1,
       0, 0, 0, 0, 1, 4, 1,
       0, 0, 0, 0, 1, 1, 8;
  std::vector<int> cliquestart{0, 1, 2, 4};
  std::vector<int> rows{3, 3, 3, 3};
  std::vector<int> cols{1, 1, 3, 3};


  Matrix temp = A;
  Matrix R(n, n);
  R.setZero();

  vector<vector<int>> root;

  SparseCholeskyDecomposition(A, cliquestart, rows, root, &R);
  EXPECT_TRUE((R*R.transpose() - A).norm() < 1e-8);
}

TEST(Cholesky, BlockDiag) {
  Matrix A;

  std::vector<Matrix> Mcliques;
  std::vector<Clique> cliques;

  //cliques.push_back({ 0, 1, 2,  7, 8});
  cliques.push_back({ 0, 1, 2});
  Mcliques.push_back(Ones(cliques.back().size()));

  //cliques.push_back({ 2, 3, 4,  7, 8});
  cliques.push_back({ 3, 4, 5});
  Mcliques.push_back(Ones(cliques.back().size()));

  //cliques.push_back({4, 5, 6, 7, 8});
  cliques.push_back({4, 5, 6 });
  Mcliques.push_back(Ones(cliques.back().size()));

  int n = GetMax(cliques) + 1;
  A = GetMatrix(n, cliques, Mcliques);

  std::vector<int> cliquestart{0, 3, 4};
  std::vector<int> size{3, 3, 3};
  std::vector<std::vector<int>> root_nodes;

  Matrix R(n, n);
  R.setZero();
  A = A + Matrix::Identity(n, n) * 10;

  SparseCholeskyDecomposition(A, cliquestart, size, root_nodes, &R);
  EXPECT_TRUE((R*R.transpose() - A).norm() < 1e-8);
}



TEST(Cholesky, TestDecomp1) {
  Matrix A;

  std::vector<Matrix> Mcliques;
  std::vector<Clique> cliques;

  cliques.push_back({0, 1, 2, 7});
  Mcliques.push_back(Ones(cliques.back().size()));

  cliques.push_back({2, 3, 4, 7, 8});
  Mcliques.push_back(Ones(cliques.back().size()));

  cliques.push_back({4, 5, 6, 7, 8});
  Mcliques.push_back(Ones(cliques.back().size()));

  int n = GetMax(cliques) + 1;
  A = GetMatrix(n, cliques, Mcliques);

  std::vector<int> cliquestart{0, 2, 4}; 
  std::vector<int> size{3, 3, 5};

  std::vector<std::vector<int>> root_nodes{{7}, {8},  {}};
  Matrix R(n, n);
  R.setZero();
  A = A + Matrix::Identity(n, n) * 100;

  SparseCholeskyDecomposition(A, cliquestart, size, root_nodes, &R);
  EXPECT_TRUE((R*R.transpose() - A).norm() < 1e-8);

  // QR = A
  // Q = A inv(R) 
  // Q Q' =  A inv(R) inv(R') A'

  // 1    1  1  1
  // 1 1     1  1
  // 1 1 1      1  = b 
}

void Append(const std::vector<int>& y, std::vector<int>* x) {
  for (const auto e : y) {
    x->push_back(e);
  }
}

TEST(Cholesky, Sizes) {
  std::vector<Matrix> Mcliques;
  std::vector<Clique> cliques;
  cliques.push_back({0, 1, 2, 6});
  Mcliques.push_back(Ones(cliques.back().size()));

  cliques.push_back({2, 3, 4, 5, 6});
  Mcliques.push_back(Ones(cliques.back().size()));

  cliques.push_back({4, 5, 6, 7, 8});
  Mcliques.push_back(Ones(cliques.back().size()));

  auto clique_order = ChordalMFS(3, cliques);

  std::vector<int> cliquestart{0, 2, 4}; 
  std::vector<int> size{3, 5, 5};

  std::vector<int> size_of_differences{2};

  std::vector<int> temp{5};
  std::vector<int> order{10};
  for (int i = 0 ; i < 2; i++) {
    DifferenceOfSorted(cliques.at(i), cliques.at(i+1), &temp);
    size_of_differences.push_back(temp.size());
  }

  std::vector<int> cliquestart2{0, 0, 0}; 
  for (unsigned int i = 1; i < cliquestart.size(); i++) {
    cliquestart2.at(i) = cliquestart2.at(i - 1) + size_of_differences.at(i - 1);
  }

  EXPECT_EQ(cliquestart2, cliquestart);

  std::vector<int> size2(cliques.size());
  int num_cliques = 3;
  for (int i = 0; i < num_cliques; i++) {
    // A < B < C
    // size A : A \ B + A \cap (B \ C)
    std::vector<int> a_minus_b{}; a_minus_b.reserve(5);
    std::vector<int> b_minus_c{}; b_minus_c.reserve(5);
    std::vector<int> a_cap_b_minus_c{}; a_cap_b_minus_c.reserve(5);

    a_minus_b = cliques.at(i);
    if (i < num_cliques - 1) {
      DifferenceOfSorted(cliques.at(i), cliques.at(i + 1), &a_minus_b);
      b_minus_c = cliques.at(i + 1);
    }

    size2.at(i) = a_minus_b.size();
    if (i < num_cliques - 2) {
      DifferenceOfSorted(cliques.at(i + 1), cliques.at(i + 2), &b_minus_c);
    }

    IntersectionOfSorted(cliques.at(i), b_minus_c, &a_cap_b_minus_c);
    size2.at(i) += a_cap_b_minus_c.size();
   }
   EXPECT_EQ(size2, size);
}

// *   *
