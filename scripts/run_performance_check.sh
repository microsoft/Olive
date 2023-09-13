#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
set -eoux pipefail

ROOT_DIR=$1
MODEL_NAME=$2
DEVICE=$3

python -m pip install -r $ROOT_DIR/.azure_pipelines/performance_check/requirements-$DEVICE.txt
python $ROOT_DIR/.azure_pipelines/performance_check/run_performance_check.py --model_name $MODEL_NAME --device $DEVICE

# clean up
rm -rf $ROOT_DIR/.azure_pipelines/performance_check/run_cache
