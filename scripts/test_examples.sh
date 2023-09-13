#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
set -eoux pipefail

ROOT_DIR=$1
EXAMPLE_FOLDER=$2
EXAMPLE_NAME=$3

# install pytest
python -m pip install pytest

# test samples
echo "Testing examples"
python -m pip install -r $ROOT_DIR/examples/$EXAMPLE_FOLDER/requirements.txt

python -m pytest -v -s --log-cli-level=WARNING --junitxml=$ROOT_DIR/logs/test_examples-TestOlive.xml $ROOT_DIR/examples/test/test_$EXAMPLE_NAME.py
