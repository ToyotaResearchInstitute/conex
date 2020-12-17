#include <vector>

namespace conex {

void PickCliqueOrder(const std::vector<std::vector<int>>& cliques_sorted,
                     int root, std::vector<int>* order,
                     std::vector<std::vector<int>>* supernodes,
                     std::vector<std::vector<int>>* separators);

}  // namespace conex
