#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
set -eoux pipefail

PIPELINE=$1
ROOT_DIR=$2
PERF_MONITORING_SCRIPT_NAME=$3

echo $PIPELINE
if [[ "$PIPELINE" == "True" ]]; then
    set +x
    source olive-venv/bin/activate
    set -x
fi

# install pytest
python -m pip install pytest

# performance monitoring
echo "performance monitoring examples"
python -m pip install -r $ROOT_DIR/perf_monitoring/requirements.txt

python -m pytest -v -s --log-cli-level=WARNING --junitxml=$ROOT_DIR/logs/performance-monitoring-TestOlive.xml $ROOT_DIR/perf_monitoring/test_$PERF_MONITORING_SCRIPT_NAME.py
