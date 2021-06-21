#include <vector>

#include "conex/tree_utils.h"

namespace conex {

void FillIn(const RootedTree& tree, int num_variables,
            const std::vector<int>& order, vector<std::vector<int>>* supernodes,
            vector<std::vector<int>>* separators);

void PickCliqueOrder(
    const std::vector<std::vector<int>>& cliques_sorted,
    const std::vector<int>& valid_leaf, int root, std::vector<int>* order,
    std::vector<std::vector<int>>* supernodes,
    std::vector<std::vector<int>>* separators,
    std::vector<std::vector<std::vector<int>>>* post_ordering = NULL);

void PickCliqueOrder(
    const std::vector<std::vector<int>>& cliques_sorted, int root,
    std::vector<int>* order, std::vector<std::vector<int>>* supernodes,
    std::vector<std::vector<int>>* separators,
    std::vector<std::vector<std::vector<int>>>* post_ordering = NULL);

}  // namespace conex
