cc_binary(
  name = "libconex.so",
  srcs = ["conex.h", "conex.cc"],
  deps = ["//conex:conex_lib"],
  copts = ["-static-libstdc++"],
  linkshared = 1,
  linkstatic = 1,
)
