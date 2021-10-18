cc_test(
    name = "interface_test",
    srcs = [
        "conex.h",
        "test/interface_test.cc",
    ],
    copts = ["-Iexternal/gtest/include"],
    linkopts = ["-lconex"],
    deps = [
        "@gtest//:main",
    ],
)

cc_test(
    name = "interface_test_soc",
    srcs = [
        "conex.h",
        "test/interface_test_soc.cc",
    ],
    copts = ["-Iexternal/gtest/include"],
    linkopts = ["-lconex"],
    deps = [
        "@gtest//:main",
    ],
)

cc_binary(
    name = "test_app",
    srcs = [
        "conex.h",
        "test/test_app.cc",
    ],
    linkopts = ["-lconex"],
)
