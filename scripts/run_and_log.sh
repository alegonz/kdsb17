#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python3 -u ${1} 2>&1 | tee ${2}
