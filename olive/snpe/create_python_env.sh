#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
set -eux

# This script creates a python 3.6 environment in $SNPE_ROOT/olive-pyenv
# and installs the required packages for SNPE-v2.18.0.240101

# Usage: ./create_python_env.sh -v/--version <python_version>
while [[ "$#" -gt 0 ]]; do
    key="$1"
    case $key in
        -v|--version)
            PY_VERSION="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

PY_ENV_NAME=olive-pyenv
FILES_DIR=$SNPE_ROOT/python-env-setup
rm -rf "$FILES_DIR"
mkdir "$FILES_DIR"

# Install conda if not already installed
if ! command -v conda; then
    # Install conda
    curl -fsSL -o "$FILES_DIR"/install_conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh "$FILES_DIR"/install_conda.sh -b -p "$FILES_DIR"/miniconda
    CONDA=$FILES_DIR/miniconda/bin/conda
else
    CONDA=conda
fi

# Create python environment
$CONDA create -y -p "$FILES_DIR"/$PY_ENV_NAME python="$PY_VERSION"

# Install snpe requirements
"$FILES_DIR"/$PY_ENV_NAME/bin/python -m pip install --upgrade pip
if [ "$PY_VERSION" == "3.6" ]; then
    "$FILES_DIR"/$PY_ENV_NAME/bin/python -m pip install onnx==1.11.0 onnx-simplifier packaging tensorflow==1.15.0 pyyaml
elif [ "$PY_VERSION" == "3.8" ]; then
    "$FILES_DIR"/$PY_ENV_NAME/bin/python -m pip install onnx onnx-simplifier packaging tensorflow pyyaml
else
    echo "Unsupported python version: $PY_VERSION, only 3.6 and 3.8 are supported"
    exit 1
fi


rm -rf "${SNPE_ROOT:?}"/$PY_ENV_NAME
mv "$FILES_DIR"/$PY_ENV_NAME "$SNPE_ROOT"/$PY_ENV_NAME

# Remove all unnecessary files
rm -rf "$FILES_DIR"
