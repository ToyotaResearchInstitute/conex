cc_test(
    name = "MyTest",
    srcs = glob(["test/test_lp.cc"]),
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

