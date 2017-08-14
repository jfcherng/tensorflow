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

UserOps=(
    "my_op.so"
    "zero_out.so"
)


###########
# compile #
###########

cd "${PROJECT_DIR}"

for UserOp in "${UserOps[@]}"
do
    bazel build -s --config opt //tensorflow/core/user_ops:"${UserOp}"
done


cd "${CURRENT_DIR}"
