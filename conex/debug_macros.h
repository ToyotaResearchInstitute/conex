#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

namespace conex {

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
  os << "\n--\n";
  for (auto e : P) {
    os << e << "\n-\n";
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

}  // namespace conex
