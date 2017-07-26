#!/usr/bin/env bash

###########
# configs #
###########

GraphOutputDir=/tmp/tensorflow

JFCHERNG_DEBUG=1
TF_CPP_MIN_VLOG_LEVEL=2 # to dump XLA graph, this should be >= 1
TF_XLA_FLAGS_ARRAY=(
    # terms:
    #     log  = show message on the screen
    #     dump = write to an external file

    # output directory
    "--tf_dump_graph_prefix=${GraphOutputDir}"
    "--xla_dump_computations_to=${GraphOutputDir}"
    "--xla_hlo_dump_graph_path=${GraphOutputDir}"

    # this flag will dump HLO graph (disregard TF_CPP_MIN_VLOG_LEVEL)
    "--xla_generate_hlo_graph=.*"

    # the format of dumped HLO IR
    # true / false = pbtxt / dot
    "--xla_hlo_dump_as_graphdef=true"

    # show HLO IR (text form) on the screen
    "--xla_log_hlo_text=.*"
)

##############
# preprocess #
##############

TF_XLA_FLAGS=${TF_XLA_FLAGS_ARRAY[*]}

mkdir -p ${GraphOutputDir}
rm -f "${GraphOutputDir}"/*

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
