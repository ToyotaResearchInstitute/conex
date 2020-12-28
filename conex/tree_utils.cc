#include "conex/tree_utils.h"
#include "assert.h"
#include <stack>
#include <vector>

using std::array;
using std::vector;

namespace conex {

vector<int> path(int x, int y, const std::vector<int>& tree,
                 const std::vector<int>& height) {
  std::vector<int> path;
  while (x != y) {
    if (height[x] < height[y]) {
      path.push_back(y);
      y = tree.at(y);
    } else {
      path.push_back(x);
      x = tree.at(x);
    }
  }
  path.push_back(x);
  return path;
}

}  // namespace conex
