#include "conex/clique_ordering.h"
#include "conex/debug_macros.h"

#include "gtest/gtest.h"

#include "conex/block_triangular_operations.h"
#include "conex/supernodal_solver.h"

namespace conex {




using Eigen::MatrixXd;
using std::vector;

std::vector<int> Union(const vector<std::vector<int>>& v) {
  auto s = v.at(0);
  for (auto vi : v) {
    std::sort(vi.begin(), vi.end());
    std::sort(s.begin(), s.end());
    s = UnionOfSorted(vi, s);
  }
  return s;
}
vector<int> LookUpSupernode(const vector<std::vector<int>>& supernodes,
                            const std::vector<int>& vin) {
  vector<int> y;
  for (auto v : vin) {
    for (int j = static_cast<int>(supernodes.size()) - 1; j >= 0; j--) {
      for (auto& sn : supernodes.at(j)) {
        if (sn == v) {
          y.push_back(j);
          break;
        }
      }
    }
  }
  return y;
}

void DoVerifyPerfectEliminationOrdering(const vector<vector<int>>& cliques_in,
                                        bool expect_fill_in = false) {
  auto cliques = cliques_in;
  Sort(&cliques);
  size_t n = cliques.size();
  vector<vector<int>> cliques_fill(cliques.size());

  // Loop over differently specified root nodes of clique tree.
  for (int root = -1; root < static_cast<int>(n); root++) {
    vector<int> order(n);
    vector<vector<int>> supernodes(n);
    vector<vector<int>> separators(n);
    PickCliqueOrder(cliques, root, &order, &supernodes, &separators);
    Sort(&supernodes);
    Sort(&cliques);
    Sort(&separators);

    for (size_t i = 0; i < n; i++) {
      if (expect_fill_in) {
        EXPECT_GE(supernodes.at(i).size() + separators.at(i).size(),
                  cliques.at(i).size());
        // Verify supernode + separator is a superset of clique
        auto intersection = cliques.at(i);
        intersection.clear();
        IntersectionOfSorted(UnionOfSorted(supernodes.at(i), separators.at(i)),
                             cliques.at(i), &intersection);
        EXPECT_EQ(intersection, cliques.at(i));
      } else {
        EXPECT_EQ(supernodes.at(i).size() + separators.at(i).size(),
                  cliques.at(i).size());
        // Verify supernode + separator  =  clique
        EXPECT_EQ(UnionOfSorted(supernodes.at(i), separators.at(i)),
                  cliques.at(i));
      }
    }
    EXPECT_EQ(Union(supernodes), Union(cliques));
  }
}

TEST(CliqueOrdering, ExpectFillIn) {
  DoVerifyPerfectEliminationOrdering({{1, 2}, {2, 3}, {3, 4}, {4, 1}},
                                     true /*expect fill-in*/);
}

TEST(CliqueOrdering, PerfectEliminationOrderFound) {
  DoVerifyPerfectEliminationOrdering(
      {{1, 2, 3, 5}, {3, 4, 5}, {4, 5, 6, 7}, {8, 9}, {1, 11}});
  DoVerifyPerfectEliminationOrdering(
      {{0, 2, 3, 5}, {3, 4, 5}, {4, 5, 6, 7}, {0, 11}});
}

TEST(CliqueOrdering, Nonmaximal) {
  DoVerifyPerfectEliminationOrdering({{0, 1}, {0, 1, 2}, {0, 1, 2, 3, 4}});
}

TEST(CliqueOrdering, SmallSize) {
  DoVerifyPerfectEliminationOrdering({{0, 1}});
  DoVerifyPerfectEliminationOrdering({{0, 1}, {1, 2}});
}

TEST(CliqueOrdering, PerfectEliminationOrderFoundDiagonal) {
  vector<vector<int>> cliques{{1}, {2}, {3}, {4}, {5}};
  DoVerifyPerfectEliminationOrdering(cliques);

  size_t n = cliques.size();
  vector<vector<int>> cliques_fill(cliques.size());

  vector<int> order(n);
  vector<vector<int>> supernodes(n);
  vector<vector<int>> seperators(n);

  PickCliqueOrder(cliques, 0, &order, &supernodes, &seperators);
  for (const auto& s : seperators) {
    EXPECT_EQ(static_cast<int>(s.size()), 0);
  }
}

TEST(CliqueOrdering, OptimalOrdering) {
  DoVerifyPerfectEliminationOrdering(
      {{1, 2, 3, 5}, {3, 4, 5}, {4, 5, 6, 7}, {8, 9}, {1, 11}});
}

} // namespace conex

