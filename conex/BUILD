cc_library(
    name = "test_util",
    srcs = glob(["test/test_util.cc"]),
    hdrs = glob([
        "test/test_util.h",
        "debug_macros.h",
    ]),
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "conex_lib",
        "@gtest//:main",
    ],
)

cc_test(
    name = "lp",
    srcs = glob(["test/test_lp.cc"]),
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "conex_lib",
        "supernodal_solver",
        "@gtest//:main",
    ],
)

cc_test(
    name = "hermitian_psd_test",
    srcs = glob(["test/hermitian_psd_test.cc"]),
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "conex_lib",
        "test_util",
        "@gtest//:main",
    ],
)

cc_test(
    name = "sdp",
    srcs = glob(["test/test_sdp.cc"]),
    copts = ["-Iexternal/gtest/include -fno-omit-frame-pointer"],
    deps = [
        "conex_lib",
        "test_util",
        "@gtest//:main",
    ],
)

cc_binary(
    name = "psdp",
    srcs = glob(["test/profile_sdp.cc"]),
    deps = [
        "conex_lib",
        "test_util",
        "@gtest//:main",
    ],
)

cc_test(
    name = "socp",
    srcs = glob(["test/test_socp.cc"]),
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "conex_lib",
        "@gtest//:main",
    ],
)

cc_test(
    name = "divergence",
    srcs = glob(["test/test_divergence.cc"]),
    copts = ["-Iexternal/gtest/include -O0"],
    deps = [
        "conex_lib",
        "@gtest//:main",
    ],
)

cc_test(
    name = "warmstart",
    srcs = glob(["test/test_warmstart.cc"]),
    copts = ["-Iexternal/gtest/include -O0"],
    deps = [
        "conex_lib",
        "test_util",
        "@gtest//:main",
    ],
)

cc_test(
    name = "approximate_eigenvalues_test",
    srcs = glob([
        "test/approximate_eigenvalues.cc",
        "debug_macros.h",
    ]),
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "approximate_eigenvalues",
        "conex_lib",
        "test_util",
        "@eigen",
        "@gtest//:main",
    ],
)

cc_library(
    name = "approximate_eigenvalues",
    srcs = ["approximate_eigenvalues.cc"],
    hdrs = ["approximate_eigenvalues.h"],
    visibility = ["//visibility:public"],
    deps = ["@eigen"],
)

cc_test(
    name = "exponential_map_pade_test",
    srcs = glob(["test/exponential_map_pade_test.cc"]),
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "exponential_map_pade",
        "@eigen",
        "@gtest//:main",
    ],
)

cc_library(
    name = "exponential_map_pade",
    srcs = [
        "debug_macros.h",
        "exponential_map_pade.cc",
    ],
    hdrs = ["exponential_map_pade.h"],
    visibility = ["//visibility:public"],
    deps = ["@eigen"],
)

cc_library(
    name = "jordan_matrix_algebra",
    srcs = [
        "debug_macros.h",
        "jordan_matrix_algebra.cc",
    ],
    hdrs = ["jordan_matrix_algebra.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
    ],
)

cc_test(
    name = "jordan_matrix_algebra_test",
    srcs = ["test/jordan_matrix_algebra_test.cc"],
    copts = ["-Iexternal/gtest/include -O0"],
    deps = [
        "jordan_matrix_algebra",
        "test_util",
        "@gtest//:main",
    ],
)

cc_test(
    name = "exponential_map_test",
    srcs = ["test/exponential_map_test.cc"],
    copts = ["-Iexternal/gtest/include -O0"],
    deps = [
        "exponential_map",
        "@gtest//:main",
    ],
)

cc_library(
    name = "exponential_map",
    srcs = ["exponential_map.cc"],
    hdrs = ["exponential_map.h"],
    visibility = ["//visibility:public"],
    deps = [
        "jordan_matrix_algebra",
        "@eigen",
    ],
)

cc_test(
    name = "supernodal_solver_test",
    srcs = ["test/supernodal_solver_test.cc"],
    copts = ["-Iexternal/gtest/include -O0"],
    deps = [
        "supernodal_solver",
        "@gtest//:main",
    ],
)

cc_test(
    name = "block_triangular_operations_test",
    srcs = ["test/block_triangular_operations_test.cc"],
    copts = ["-Iexternal/gtest/include -O0"],
    deps = [
        "supernodal_solver",
        "@gtest//:main",
    ],
)

cc_test(
    name = "clique_ordering_test",
    srcs = ["test/clique_ordering_test.cc"],
    copts = ["-Iexternal/gtest/include -O0"],
    deps = [
        "supernodal_solver",
        "@gtest//:main",
    ],
)

cc_test(
    name = "assembly_test",
    srcs = ["test/assembly_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "supernodal_solver",
        "@gtest//:main",
    ],
)

cc_test(
    name = "tree_utils_test",
    srcs = ["test/tree_utils_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "supernodal_solver",
        "@gtest//:main",
    ],
)

cc_library(
    name = "supernodal_solver",
    srcs = [
        "block_triangular_operations.cc",
        "clique_ordering.cc",
        "debug_macros.h",
        "kkt_assembler.cc",
        "kkt_solver.cc",
        "supernodal_solver.cc",
        "tree_utils.cc",
        "triangular_matrix_workspace.cc",
    ],
    hdrs = [
        "block_triangular_operations.h",
        "clique_ordering.h",
        "constraint.h",
        "constraint_manager.h",
        "debug_macros.h",
        "equality_constraint.h",
        "error_checking_macros.h",
        "error_codes.h",
        "kkt_assembler.h",
        "kkt_solver.h",
        "kkt_system_assembler.h",
        "linear_workspace.h",
        "memory_utils.h",
        "newton_step.h",
        "supernodal_cholesky_data.h",
        "supernodal_solver.h",
        "tree_utils.h",
        "triangular_matrix_workspace.h",
        "workspace.h",
    ],
    visibility = ["//visibility:public"],
    deps = ["@eigen"],
)

cc_library(
    name = "conex_lib",
    srcs = [
        "cone_program.cc",
        "dense_lmi_constraint.cc",
        "divergence.cc",
        "hermitian_psd.cc",
        "linear_constraint.cc",
        "psd_constraint.cc",
        "quadratic_cone_constraint.cc",
        "soc_constraint.cc",
    ],
    hdrs = [
        "cone_program.h",
        "constraint.h",
        "debug_macros.h",
        "dense_lmi_constraint.h",
        "divergence.h",
        "error_checking_macros.h",
        "error_codes.h",
        "hermitian_psd.h",
        "linear_constraint.h",
        "memory_utils.h",
        "newton_step.h",
        "psd_constraint.h",
        "quadratic_cone_constraint.h",
        "soc_constraint.h",
        "workspace.h",
        "workspace_soc.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "approximate_eigenvalues",
        "exponential_map",
        "exponential_map_pade",
        "jordan_matrix_algebra",
        "supernodal_solver",
        "@eigen",
    ],
)

cc_test(
    name = "equality_constraints_test",
    srcs = glob(["test/equality_constraints_test.cc"]),
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "conex_lib",
        "@gtest//:main",
    ],
)

cc_binary(
    name = "solver_failures",
    srcs = ["test/solver_failures.cc"],
    deps = [
        "conex_lib",
        "@gtest//:main",
    ],
)
