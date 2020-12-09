#include <iostream>
#include "conex/error_codes.h"

// TODO(FrankPermenter): return conex error codes on failure.

#if 0
#define CONEX_DEMAND(x, msg)                          \
  if (!(x)) {                                         \
    std::cout << "Conex error: " << msg << std::endl; \
    return 1;                                         \
  }
#else
#define CONEX_DEMAND(x, msg)                                                   \
  if (!(x)) {                                                                  \
    std::cout << __FILE__ << " line " << __LINE__ << ": " << msg << std::endl; \
    return 1;                                                                  \
  }
#endif
