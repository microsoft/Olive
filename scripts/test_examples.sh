#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
set -eoux pipefail

PIPELINE=$1
ROOT_DIR=$2
EXAMPLE=$3

echo $PIPELINE
if [[ "$PIPELINE" == "True" ]]; then
    set +x
    source olive-venv/bin/activate
    set -x
fi

# install pytest
python -m pip install pytest

# test samples
echo "Testing examples"
python -m pip install -r $ROOT_DIR/examples/$EXAMPLE/requirements.txt

# TODO: need to remove later
echo "Installing custom packages for whisper"
if [[ "$EXAMPLE" == "whisper" ]]; then
    python -m pip uninstall -y onnxruntime onnxruntime-extensions
    python -m pip install ort-nightly==1.15.0.dev20230429003 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
    export OCOS_NO_OPENCV=1
    python -m pip install git+https://github.com/microsoft/onnxruntime-extensions.git
fi

python -m pytest -v -s --log-cli-level=WARNING --junitxml=$ROOT_DIR/logs/test_examples-TestOlive.xml $ROOT_DIR/examples/test_$EXAMPLE.py
