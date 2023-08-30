#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
set -eoux pipefail

PIPELINE=$1
ROOT_DIR=$2
MODEL_NAME=$3
DEVICE=$4

echo $PIPELINE
if [[ "$PIPELINE" == "True" ]]; then
    set +x
    source olive-venv/bin/activate
    set -x
fi

python -m pip install -r $ROOT_DIR/.azure_pipelines/performance_check/requirements-$DEVICE.txt
python $ROOT_DIR/.azure_pipelines/performance_check/run_performance_check.py --model_name $MODEL_NAME --device $DEVICE

# clean up
rm -rf $ROOT_DIR/.azure_pipelines/performance_check/run_cache