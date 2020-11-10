cc_library(
    name = "libconex",
    hdrs = ["conex.h"],
    srcs = ["conex.cc"],
    visibility = ["//visibility:public"],
    deps = ["@eigen", "//conex:conex_lib"], 
)

cc_test(
    name = "interface_test",
    srcs = ["test/interface_test.cc"],
    copts = ["-Iexternal/gtest/include"], 
    deps = ["@gtest//:main", "libconex"],
)

