#!/usr/bin/env bash

###########
# configs #
###########

GraphOutputDir=/tmp/tensorflow

JFCHERNG_DEBUG=1
TF_CPP_MIN_VLOG_LEVEL=2 # to dump XLA graph, this should be >= 1
TF_XLA_FLAGS_ARRAY=(
    "--tf_dump_graph_prefix=${GraphOutputDir}"
    "--xla_hlo_dump_graph_path=${GraphOutputDir}"
    "--xla_dump_computations_to=${GraphOutputDir}"
    "--xla_generate_hlo_graph=.*"
    "--xla_log_hlo_text=.*"
    "--xla_hlo_dump_as_graphdef=1"
    "--xla_hlo_profile=1"
)

##############
# preprocess #
##############

TF_XLA_FLAGS=${TF_XLA_FLAGS_ARRAY[*]}

mkdir -p ${GraphOutputDir}

##############
# export ENV #
##############

export JFCHERNG_DEBUG="${JFCHERNG_DEBUG}"
export TF_CPP_MIN_VLOG_LEVEL="${TF_CPP_MIN_VLOG_LEVEL}"
export TF_XLA_FLAGS="${TF_XLA_FLAGS}"

###########
# execute #
###########

python3 "$@"
