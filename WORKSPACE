workspace(name = "conex")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

#http_archive(
#    name = "eigen",
#    build_file = "//third_party:eigen.BUILD",
#    url = "https://bitbucket.org/eigen/eigen/get/323c052e1731.tar.bz2",
#    sha256 = "9f13cf90dedbe3e52a19f43000d71fdf72e986beb9a5436dddcd61ff9d77a3ce",
#    strip_prefix = "eigen-eigen-323c052e1731",
#)
# eigen
http_archive(
    name = "eigen",
    build_file = "//third_party:eigen.BUILD",
    sha256 = "d956415d784fa4e42b6a2a45c32556d6aec9d0a3d8ef48baee2522ab762556a9",
    url = "https://mirror.bazel.build/bitbucket.org/eigen/eigen/get/fd6845384b86.tar.gz",
    strip_prefix="eigen-eigen-fd6845384b86"
)
#http_archive(
#    name = "eigen",
#    strip_prefix = "eigen-3.3.7",
#    sha256 = "d56fbad95abf993f8af608484729e3d87ef611dd85b3380a8bad1d5cbc373a57",
#    urls = [
#        "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz"
#    ],
#    build_file = "//third_party:eigen.BUILD"
#)

http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "//third_party:gtest.BUILD",
    strip_prefix = "googletest-release-1.7.0",
)
