#!/usr/bin/env bash

####################################################################
# note that you have to run ./configure before running this script #
####################################################################

# some configurations
TENSORFLOW_BUILD_DIR="/tmp/tensorflow_pkg"
PIP_EXECUTABLE="pip3"
PIP_PACKAGE_NAME="tensorflow"

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

# go to the root dir of this project
cd "${PROJECT_DIR}" || exit

# remove old builds
# ":?" means If parameter is null or unset, the expansion of word
# (or a message to that effect if word is not present) is written
# to the standard error and the shell, if it is not interactive, exits.
# Otherwise, the value of parameter is substituted.
rm -f ${TENSORFLOW_BUILD_DIR:?}/*.whl

# build
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package || exit
bazel-bin/tensorflow/tools/pip_package/build_pip_package ${TENSORFLOW_BUILD_DIR} || exit

# remove previously installed TensorFlow
if ( ${PIP_EXECUTABLE} list --format=legacy | grep ${PIP_PACKAGE_NAME} >/dev/null ); then
    sudo -H ${PIP_EXECUTABLE} uninstall -y ${PIP_PACKAGE_NAME}
fi

# install
sudo -H ${PIP_EXECUTABLE} install ${TENSORFLOW_BUILD_DIR}/*.whl || exit

# go back to current dir
cd "${CURRENT_DIR}" || exit
