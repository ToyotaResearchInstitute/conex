#pragma once
#include <numeric>
#include <vector>

#include <Eigen/Dense>

#include "conex/debug_macros.h"
using Clique = std::vector<int>;

struct TriangularMatrixWorkspace {
  TriangularMatrixWorkspace(const std::vector<Clique>& path_, 
                            const std::vector<int>& supernode_size_) :
                            supernode_size(supernode_size_) {
      separators.resize(path_.size());
      int cnt = 0;
      for (auto& si : separators) {
        for (size_t i = supernode_size.at(cnt); i < path_.at(cnt).size(); i++) {
          si.push_back(path_.at(cnt).at(i));
        }
        cnt++;
      } 

      cnt = 0;
      snodes.resize(path_.size());
      for (auto& si : snodes) {
        for (int i = 0; i < supernode_size.at(cnt); i++) {
          si.push_back(path_.at(cnt).at(i));
        }
        cnt++;
      } 

      N = std::accumulate(supernode_size.begin(), supernode_size.end(), 0);
    }

  TriangularMatrixWorkspace(const std::vector<Clique>& snodes_,
      const std::vector<Clique>& separators_) :
  snodes(snodes_), separators(separators_) {

    int cnt = 0;
    supernode_size.resize(snodes.size());
    for (auto& si : supernode_size) {
      si = snodes.at(cnt).size();
      cnt++;
    } 
    N = std::accumulate(supernode_size.begin(), supernode_size.end(), 0);
  }

  int N;
  // TODO(FrankPermenter): Remove all of these members.
  std::vector<int> supernode_size;
  std::vector<Eigen::Map<Eigen::MatrixXd>> diagonal;
  std::vector<Eigen::Map<Eigen::MatrixXd>> off_diagonal;
  std::vector<std::vector<double*>> seperator_diagonal;

  std::vector<std::vector<int>> separators;
  std::vector<std::vector<int>> snodes;

  int SizeOfSupernode(int i) const {
    return supernode_size.at(i) * supernode_size.at(i);
  }

  int SizeOfSeparator(int i) const {
    return supernode_size.at(i) * separators.at(i).size(); 
  }

  friend int SizeOf(const TriangularMatrixWorkspace& o) {
    int size = 0;
    for (size_t j = 0; j < o.snodes.size(); j++) {
      size += o.SizeOfSupernode(j);
    }
    for (size_t j = 0; j < o.separators.size(); j++) {
      size += o.SizeOfSeparator(j);
    }
    return size;
  }

  friend void Initialize(TriangularMatrixWorkspace* o, double* data_start)  {
    double*  data = data_start;
    for (size_t j = 0; j < o->snodes.size(); j++) {
      o->diagonal.emplace_back(data, o->supernode_size.at(j), o->supernode_size.at(j));
      Eigen::Map<Eigen::MatrixXd> test(data, o->supernode_size.at(j), o->supernode_size.at(j));
      data += o->SizeOfSupernode(j);
      o->off_diagonal.emplace_back(data, o->supernode_size.at(j), 
                                o->separators.at(j).size());
      data += o->SizeOfSeparator(j);
    }
    for (size_t j = 0; j < o->snodes.size(); j++) {
      o->seperator_diagonal.push_back(o->S_S(j));
    }
  }
 private:
  // TODO(FrankPermenter): Remove this method.
  std::vector<double*> S_S(int clique);
};
