cc_binary(
  name = "libconex.so",
  srcs = ["conex.cc"],
  deps = ["//conex:conex_lib", "api"],
  copts = ["-static-libstdc++"],
  linkshared = 1,
  linkstatic = 1,
)

cc_library(
    name = "api",
    hdrs = [ "conex.h" ],
    visibility = ["//visibility:public"],
)
