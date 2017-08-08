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

# external dlsym() for XLA custom-call
CustomCalls=(
    "relu_op_jfcherng_xla_impl.so"
)


###########
# compile #
###########

cd "${PROJECT_DIR}"

for CustomCall in "${CustomCalls[@]}"
do
    bazel build -s --config=opt //tensorflow/compiler/tf2xla/kernels:"${CustomCall}"
done


cd "${CURRENT_DIR}"
