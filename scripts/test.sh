#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
set -eoux pipefail

PIPELINE=$1
ROOT_DIR=$2
TEST_TYPE=$3

echo "Running tests in $TEST_TYPE"

# install pytest
python -m pip install pytest
python -m pip install -r $ROOT_DIR/test/requirements-test.txt

# run tests
coverage run --source=$ROOT_DIR/olive -m pytest -v -s --log-cli-level=WARNING --junitxml=$ROOT_DIR/logs/test-TestOlive.xml $ROOT_DIR/test/$TEST_TYPE
coverage xml
