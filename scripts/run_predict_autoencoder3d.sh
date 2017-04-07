#!/usr/bin/env bash

PYTHONHASHSEED=0 \
THEANO_FLAGS='dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic' \
python3 -u predict_autoencoder3d.py ${1} ${2}