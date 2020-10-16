#pragma once
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#   define DUMP(x) \
    do { \
            std::cerr << __FILE__  << " line " << __LINE__ \
                      << std::endl << #x ":" << std::endl << x \
                      << std::endl;\
        } while (false)

template<typename T1, typename T2>
inline std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& P) {
  os << P.first << " " << P.second << "\n\n";
  return os;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& P) {
  for (auto e : P) {
    os << e << "\n\n";
  }
  return os;
}

template<typename T, int size>
inline std::ostream& operator<<(std::ostream& os, const std::array<T, size>& P) {
  for (auto e : P) {
    os << e << "\n\n";
  }
  return os;
}


#   define LOG(x) \
    do { \
            std::cout  << #x ":" << " " << x << ", "; \
        } while (false)

#   define REPORT(x) \
    do { \
            std::cout << #x ":" << " " << x << ", "; \
        } while (false)


#   define PRINTSTATUS(x) \
    do { \
            std::cout << "Status: " << x << "\n\n"; \
        } while (false)
