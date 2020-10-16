cc_test(
    name = "lp",
    srcs = glob(["test/test_lp.cc"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "conex_lib"],
)

cc_test(
    name = "sdp",
    srcs = glob(["test/test_util.h", "test/test_util.cc", "test/test_sdp.cc"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "conex_lib"],
)

cc_test(
    name = "socp",
    srcs = glob(["test/test_socp.cc"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "conex_lib"],
)

cc_test(
    name = "octo",
    srcs = glob(["test/test_octonions.cc"]),
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "conex_lib"],
)


cc_library(
    name = "conex_lib",
    hdrs = glob([ "*.h" ]),
    srcs =  glob(["*.cc"]),
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
    ],
)

