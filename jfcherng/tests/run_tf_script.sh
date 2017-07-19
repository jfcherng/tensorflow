#!/usr/bin/env bash

# configs
GraphOutputDir=/tmp/tensorflow

JFCHERNG_DEBUG=1
TF_CPP_MIN_VLOG_LEVEL=1 # to dump XLA graph, this should be >= 1
TF_XLA_FLAGS="--tf_dump_graph_prefix=${GraphOutputDir} --xla_hlo_dump_graph_path=${GraphOutputDir} --xla_generate_hlo_graph=.*"

# execute
TF_XLA_FLAGS="${TF_XLA_FLAGS}" \
JFCHERNG_DEBUG="${JFCHERNG_DEBUG}" \
TF_CPP_MIN_VLOG_LEVEL="${TF_CPP_MIN_VLOG_LEVEL}" \
python3 "$@"
