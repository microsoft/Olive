#!/bin/bash
# Script to install dependencies and run tests
# $1: PyTorch package name or version (e.g., torch or torch==1.12.0)
# $2: ONNX Runtime package name or version (e.g., onnxruntime or onnxruntime==1.20.1)
# $3: Whether to use ONNX Runtime nightly (true/false)
# $4: Path to the requirements.txt file
# $5: Path to the test file to run
# $6: Whether to use coverage tracking (true/false)

# activate venv
source olive-venv/bin/activate

# Step 1: Install PyTorch
pip install "$1"

# Step 2: Install ONNX Runtime (Nightly or Stable)
if [ "$3" = "true" ]; then
    echo "Installing ONNX Runtime Nightly..."
    pip install -r  https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt
    pip install "$2" --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --disable-pip-version-check
else
    echo "Installing ONNX Runtime Stable..."
    pip install "$2"
fi

# Step 3: Install additional dependencies
echo "Installing additional dependencies..."
pip install pytest azure-identity azure-storage-blob tabulate
pip install -r "$4"

# Step 4: Run tests with or without coverage tracking
if [ "$6" = "true" ]; then
    echo "Running tests with coverage tracking..."
    coverage run -m pytest -vv -s --junitxml=/logs/test_examples-TestOlive.xml "$5"
    coverage xml -o /logs/coverage.xml
else
    echo "Running tests without coverage tracking..."
    python -m pytest -vv -s --junitxml=/logs/test_examples-TestOlive.xml "$5"
fi
