#!/bin/bash
# Script to install dependencies and run tests
# $1: PyTorch package name or version (e.g., torch or torch==1.12.0)
# $2: ONNX Runtime package name or version (e.g., onnxruntime or onnxruntime==1.20.1)
# $3: Whether to use ONNX Runtime nightly (true/false)
# $4: Path to the requirements.txt file
# $5: Path to the test file to run
# $6: Whether to use coverage tracking (true/false)
# $7: HF Token

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

# Set HF Token
pip install huggingface-hub
hf auth login --token "$7"

pip list

# Step 4: Run tests with or without coverage tracking
XML_PATH="/logs/TestOlive.xml"
if [ "$6" = "true" ]; then
    echo "Running tests with coverage tracking..."
    coverage run -m pytest -vv -s --junitxml="$XML_PATH" -p no:warnings --disable-warnings --log-cli-level=WARNING "$5"
    coverage xml -o /logs/coverage.xml
else
    echo "Starting pytest at $(date)"
    echo "Running tests without coverage tracking..."
    timeout 1100 python -m pytest -vv -s --junitxml="$XML_PATH" -p no:warnings --disable-warnings --log-cli-level=WARNING "$5"
    exit_code=$?
    echo "pytest exited with code $exit_code"

    # Handle timeout cases
    if [[ $exit_code -eq 124 ]]; then
        if [[ -f "$XML_PATH" ]]; then
            echo "Timed out but test results XML found. Success."
            exit 0
        else
            echo "Timed out and no test result XML. Failure."
            exit 1
        fi
    fi

    exit $exit_code
fi
