cc_library(
    name = "libconex",
    srcs = ["conex.cc"],
    hdrs = ["conex.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//conex:conex_lib",
        "@eigen",
    ],
)

cc_test(
    name = "interface_test",
    srcs = ["test/interface_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "libconex",
        "@gtest//:main",
    ],
)

cc_test(
    name = "interface_test_soc",
    srcs = ["test/interface_test_soc.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "libconex",
        "@gtest//:main",
    ],
)
