# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
set -eux

FILES_DIR=$SNPE_ROOT/python36-env-setup
rm -rf $FILES_DIR
mkdir $FILES_DIR

# Install conda if not already installed
if ! command -v conda; then
    # Install conda
    curl -fsSL -o $FILES_DIR/install_conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh $FILES_DIR/install_conda.sh -b -p $FILES_DIR/miniconda
    CONDA=$FILES_DIR/miniconda/bin/conda
else
    CONDA=conda
fi

# Create python 3.6 environment
$CONDA create -y -p $FILES_DIR/python36-env python=3.6

# Install snpe requirements
$FILES_DIR/python36-env/bin/python -m pip install --upgrade pip
$FILES_DIR/python36-env/bin/python -m pip install onnx==1.11.0 onnx-simplifier packaging tensorflow==1.15.0 pyyaml

# move the python36-env to the correct location
rm -rf $SNPE_ROOT/python36-env
mv $FILES_DIR/python36-env $SNPE_ROOT/python36-env

# Remove all unnecessary files
rm -rf $FILES_DIR
