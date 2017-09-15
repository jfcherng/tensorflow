#!/usr/bin/env bash

####################################################################
# note that you have to run ./configure before running this script #
####################################################################

# some configurations
TENSORFLOW_BUILD_DIR="/tmp/tensorflow_pkg"
PIP_EXECUTABLE="pip3"
PIP_PACKAGE_NAME="tensorflow"
PIP_DEPS=( numpy six wheel )

# do not modify these
BUILD_WITH_GPU=true
TF_CONFIGS=( --config=opt )

# auto generated constants
CURRENT_DIR="$( pwd )"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR=$( readlink -m "${SCRIPT_DIR:?}/.." )
PIP_PACKAGE_LIST=$( ${PIP_EXECUTABLE} list --format=legacy )

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

echo ""
echo "# config starts"
echo ""

# build with GPU?
read -rp "Build TensorFlow with GPU support? [Y/n] " answer
if [ "${answer,,}" = "n" ]; then
    BUILD_WITH_GPU=false
fi

echo ""
echo "# config ends"
echo ""

# install pip dependencies
for pip_dep in ${PIP_DEPS[*]}; do
    pip_package_found=$( echo "${PIP_PACKAGE_LIST}" | grep "${pip_dep} " )
    if [[ -z $pip_package_found ]]; then
        yes | sudo -H ${PIP_EXECUTABLE} install "${pip_dep}"
    fi
done

# build
if [ "${BUILD_WITH_GPU}" = "true" ]; then
    TF_CONFIGS+=( --config=cuda )
fi
echo "[INFO] Build TensorFlow with configs: ${TF_CONFIGS[*]}"
bazel build "${TF_CONFIGS[@]}" //tensorflow/tools/pip_package:build_pip_package || exit
bazel-bin/tensorflow/tools/pip_package/build_pip_package "${TENSORFLOW_BUILD_DIR}" || exit

# remove previously installed TensorFlow
pip_package_found=$( echo "${PIP_PACKAGE_LIST}" | grep "${PIP_PACKAGE_NAME} " )
if [[ -n $pip_package_found ]]; then
    yes | sudo -H ${PIP_EXECUTABLE} uninstall "${PIP_PACKAGE_NAME}"
fi

# install TensorFlow
sudo -H ${PIP_EXECUTABLE} install ${TENSORFLOW_BUILD_DIR}/*.whl || exit

# go back to current dir
cd "${CURRENT_DIR}" || exit
