
cc_library(
    name = "test_util",
    srcs = glob(["test/test_util.cc"]),
    hdrs = glob(["test/test_util.h", "debug_macros.h"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "conex_lib"],
)

cc_test(
    name = "lp",
    srcs = glob(["test/test_lp.cc"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "conex_lib"],
)

cc_test(
    name = "hermitian",
    srcs = glob(["test/test_hermitian.cc"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "conex_lib", "test_util"],
)

cc_test(
    name = "sdp",
    srcs = glob(["test/test_sdp.cc"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "conex_lib", "test_util"],
)

cc_test(
    name = "socp",
    srcs = glob(["test/test_socp.cc"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "conex_lib"],
)


cc_test(
    name = "division_algebras",
    srcs = glob(["test/test_division_algebras.cc", "jordan_matrix_algebra.h", "jordan_matrix_algebra.cc"]),
    copts = ["-Iexternal/gtest/include -O0"], 
    deps = ["@gtest//:main", "conex_lib"],
)

cc_test(
    name = "divergence",
    srcs = glob(["test/test_divergence.cc"]),
    copts = ["-Iexternal/gtest/include -O0"], 
    deps = ["@gtest//:main", "conex_lib"],
)

cc_test(
    name = "warmstart",
    srcs = glob(["test/test_warmstart.cc"]),
    copts = ["-Iexternal/gtest/include -O0"], 
    deps = ["@gtest//:main", "conex_lib", "test_util"],
)

cc_library(
    name = "cholesky",
    srcs = glob(["debug_macros.h", "test/util.h", "cholesky_decomposition.h", "cholesky_decomposition.cc", "elimination_ordering.cc", "elimination_ordering.h"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@eigen"],
)

cc_test(
    name = "cholesky_test",
    srcs = glob(["test/cholesky_test.cc"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "@eigen", "cholesky"],
)

cc_test(
    name = "approximate_eigenvalues",
    srcs = glob(["test/approximate_eigenvalues.cc", "debug_macros.h"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "@eigen", "conex_lib"],
)

cc_library(
    name = "conex_lib",
    hdrs = glob([ "*.h" ]),
    srcs = glob(["*.cc"]),
    copts = ["-O3", "-DEIGEN_USE_BLAS"],
    #copts = ["-O3"],
    visibility = ["//visibility:public"],
    deps = [ "@eigen" ],
)
