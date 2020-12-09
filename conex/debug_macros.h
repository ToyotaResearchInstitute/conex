#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#define DUMP(x)                                                \
  do {                                                         \
    std::cerr << __FILE__ << " line " << __LINE__ << std::endl \
              << #x ":" << std::endl                           \
              << x << std::endl;                               \
  } while (false)

#define START_TIMER(x)                                       \
  {                                                          \
    auto start1 = std::chrono::high_resolution_clock::now(); \
    std::cout << #x << "(us)"                                \
              << ":";

#define END_TIMER                                                            \
  auto stop1 = std::chrono::high_resolution_clock::now();                    \
  std::cout << " "                                                           \
            << std::chrono::duration_cast<std::chrono::microseconds>(stop1 - \
                                                                     start1) \
                   .count()                                                  \
            << ", ";                                                         \
  }

template <typename T1, typename T2>
inline std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& P) {
  os << P.first << " " << P.second << "\n\n";
  return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& P) {
  for (auto e : P) {
    os << e << "\n\n";
  }
  return os;
}

template <typename T, int size>
inline std::ostream& operator<<(std::ostream& os,
                                const std::array<T, size>& P) {
  for (auto e : P) {
    os << e << "\n\n";
  }
  return os;
}

template <typename T>
Eigen::MatrixXd Sparsity(const T& x) {
  Eigen::MatrixXd y(x.rows(), x.cols());
  for (int i = 0; i < x.rows(); i++) {
    for (int j = 0; j < x.cols(); j++) {
      if (std::fabs(x(i, j)) > 1e-9) {
        y(i, j) = 1;
      } else {
        y(i, j) = 0;
      }
    }
  }
  return y;
}

#define LOG(x)                     \
  do {                             \
    std::cout << #x ":"            \
              << " " << x << ", "; \
  } while (false)

#define REPORT(x)                  \
  do {                             \
    std::cout << #x ":"            \
              << " " << x << ", "; \
  } while (false)

#define PRINTSTATUS(x)                      \
  do {                                      \
    std::cout << "Status: " << x << "\n\n"; \
  } while (false)
