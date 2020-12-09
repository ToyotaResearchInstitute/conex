#pragma once

namespace conex {
constexpr unsigned int get_size_aligned(unsigned int n) {
  // Make a multiple of 4.
  int remainder = n % 4;
  if (remainder != 0) {
    return n + 4 - remainder;
  } else {
    return n;
  }
}
} // namespace conex
