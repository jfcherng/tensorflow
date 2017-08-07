#!/usr/bin/env bash

# some constants
CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR=$(readlink -m "${SCRIPT_DIR:?}/..")

# debug
cat << EOF
CURRENT_DIR = ${CURRENT_DIR}
SCRIPT_DIR  = ${SCRIPT_DIR}
PROJECT_DIR = ${PROJECT_DIR}
EOF


###########
# compile #
###########

cd "${PROJECT_DIR}"

bazel build -s --config=opt //tensorflow/compiler/tf2xla/kernels:*.so

cd "${CURRENT_DIR}"
