# Description:
# C API for TensorFlow, for use by client language bindings.

licenses(["notice"])  # Apache 2.0

# -----------------------------------------------------------------------------
# Public targets

cc_library(
    name = "version",
    hdrs = ["version.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    deps = [],
    visibility = ["//visibility:public"],
)

# -----------------------------------------------------------------------------
# Google-internal targets.

filegroup(
    name = "hdrs_files",
    srcs = glob(
        ["**/*.h"],
        exclude = [],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "srcs_files",
    srcs = glob(
        ["**/*.cc"],
        exclude = [],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
