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
    name = "hermitian_psd_test",
    srcs = glob(["test/hermitian_psd_test.cc"]),
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
    name = "approximate_eigenvalues_test",
    srcs = glob(["test/approximate_eigenvalues.cc", "debug_macros.h"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "@eigen", "approximate_eigenvalues", "conex_lib"],
)

cc_library(
    name = "approximate_eigenvalues",
    hdrs = [ "approximate_eigenvalues.h"],
    srcs = ["approximate_eigenvalues.cc"],
    visibility = ["//visibility:public"],
    deps = [ "@eigen" ],
)

cc_test(
    name = "matrix_exponential_test",
    srcs = glob(["test/matrix_exponential_test.cc"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "@eigen", "matrix_exponential"],
)

cc_library(
    name = "matrix_exponential",
    hdrs = [ "matrix_exponential.h"],
    srcs = ["matrix_exponential.cc", "debug_macros.h"],
    visibility = ["//visibility:public"],
    deps = [ "@eigen" ],
)

cc_library(
    name = "jordan_matrix_algebra",
    hdrs = [ "jordan_matrix_algebra.h"],
    srcs = ["jordan_matrix_algebra.cc", "debug_macros.h"],
    visibility = ["//visibility:public"],
    deps = [ "@eigen", "eigen_decomp" ],
)

cc_test(
    name = "jordan_matrix_algebra_test",
    srcs = ["test/jordan_matrix_algebra_test.cc"],
    copts = ["-Iexternal/gtest/include -O0"], 
    deps = ["@gtest//:main", "jordan_matrix_algebra", "conex_lib"],
)

cc_library(
    name = "vect_jordan_matrix_algebra",
    hdrs = [ "vect_jordan_matrix_algebra.h"],
    srcs = ["vect_jordan_matrix_algebra.cc", "debug_macros.h"],
    visibility = ["//visibility:public"],
    deps = [ "@eigen", "eigen_decomp" ],
)

cc_test(
    name = "vect_jordan_matrix_algebra_test",
    srcs = ["test/vect_jordan_matrix_algebra_test.cc"],
    copts = ["-Iexternal/gtest/include -O0"], 
    deps = ["@gtest//:main", "vect_jordan_matrix_algebra"],
)
cc_library(
    name = "eigen_decomp",
    hdrs = [ "eigen_decomp.h"],
    srcs = ["eigen_decomp.cc", "eigen_decomp.h"],
    visibility = ["//visibility:public"],
    deps = [ "@eigen" ],
)

cc_test(
    name = "exponential_map_test",
    srcs = ["test/exponential_map_test.cc"],
    copts = ["-Iexternal/gtest/include -O0"], 
    deps = ["@gtest//:main", "exponential_map"],
)
cc_library(
    name = "exponential_map",
    hdrs = ["exponential_map.h"],
    srcs = ["exponential_map.cc"],
    visibility = ["//visibility:public"],
    deps = [ "@eigen", "vect_jordan_matrix_algebra" ],
)

cc_library(
    name = "conex_lib",
    hdrs = ["cholesky_decomposition.h",
            "cone_program.h",
            "constraint.h",
            "debug_macros.h",
            "dense_lmi_constraint.h",
            "divergence.h",
            "elimination_ordering.h",
            "hermitian_psd.h",
            "linear_constraint.h",
            "memory_utils.h",
            "newton_step.h",
            "psd_constraint.h",
            "quadratic_cone_constraint.h",
            "soc_constraint.h",
            "workspace.h",
            "workspace_soc.h"],
    srcs = ["cholesky_decomposition.cc",
            "cone_program.cc",
            "dense_lmi_constraint.cc",
            "divergence.cc",
            "elimination_ordering.cc",
            "hermitian_psd.cc",
            "linear_constraint.cc",
            "psd_constraint.cc",
            "quadratic_cone_constraint.cc",
            "soc_constraint.cc"],
    visibility = ["//visibility:public"],
    deps = [ "@eigen", "approximate_eigenvalues", "matrix_exponential", "jordan_matrix_algebra", "eigen_decomp" ],
)


