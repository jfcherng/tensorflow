#!/usr/bin/env bash

# configs
JFCHERNG_DEBUG=1
TF_CPP_MIN_VLOG_LEVEL=0 # to dump XLA graph, this should be >= 1
TF_XLA_FLAGS="--tf_dump_graph_prefix=/tmp/tensorflow --xla_generate_hlo_graph=.*"

TF_XLA_FLAGS="${TF_XLA_FLAGS}" \
JFCHERNG_DEBUG="${JFCHERNG_DEBUG}" \
TF_CPP_MIN_VLOG_LEVEL="${TF_CPP_MIN_VLOG_LEVEL}" \
python3 "$@"
