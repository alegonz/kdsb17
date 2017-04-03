#!/usr/bin/env bash

PYTHONHASHSEED=0
THEANO_FLAGS='dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic'

python3 -u train.py 2>&1 | tee training.log