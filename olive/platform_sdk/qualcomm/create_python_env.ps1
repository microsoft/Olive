# This script creates a python environment in $QNN_SDK_ROOT\olive-pyenv or $SNPE_ROOT\olive-pyenv
# e.g. create_python_env.ps1 -py_version 3.6 -sdk snpe

param (
    [string]$PY_VERSION,
    [string]$SDK
)

$ErrorActionPreference = "Stop"

if ($SDK -eq "snpe") {
    $SDK_ROOT = $env:SNPE_ROOT
}
elseif ($SDK -eq "qnn") {
    $SDK_ROOT = $env:QNN_SDK_ROOT
}
else {
    Write-Host "Unknown SDK: $SDK"
    exit 1
}

$PY_ENV_NAME = "olive-pyenv"
$FILES_DIR = Join-Path $SDK_ROOT "python-env-setup"
Remove-Item -Path $FILES_DIR -Recurse -Force -ErrorAction SilentlyContinue
New-Item -Path $FILES_DIR -ItemType Directory | Out-Null

# Install conda if not already installed
if (!(Get-Command -Name "conda" -ErrorAction SilentlyContinue)) {
    # Install conda
    $CONDA_INSTALLER = Join-Path $FILES_DIR "Miniconda3-latest-Windows-x86_64.exe"
    Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $CONDA_INSTALLER
    Start-Process -FilePath $CONDA_INSTALLER -ArgumentList "/S /D=$FILES_DIR\miniconda" -Wait
    $CONDA = Join-Path $FILES_DIR "miniconda\Scripts\conda.exe"
}
else {
    $CONDA = "conda"
}

# Create python environment
& $CONDA create -y -p (Join-Path $FILES_DIR $PY_ENV_NAME) python=$PY_VERSION

# Install snpe requirements
# if PIP_EXTRA_ARGS is set, then use it else use ""
$PIP_EXTRA_ARGS = $env:PIP_EXTRA_ARGS
# to back-compat with older versions of the powershell
$PYTHON_PATH = Join-Path (Join-Path $FILES_DIR $PY_ENV_NAME) "python.exe"
& $PYTHON_PATH -m pip install --upgrade pip $PIP_EXTRA_ARGS
& $PYTHON_PATH -m pip install --upgrade pip $PIP_EXTRA_ARGS
if ($PY_VERSION -eq "3.6") {
    & $PYTHON_PATH -m pip install onnx==1.11.0 onnx-simplifier packaging tensorflow==1.15.0 pyyaml pandas==1.1.5 numpy==1.18.5 $PIP_EXTRA_ARGS
}
elseif ($PY_VERSION -eq "3.8") {
    & $PYTHON_PATH -m pip install onnx onnx-simplifier packaging tensorflow==2.10.1 pyyaml pandas==1.1.5 numpy==1.23.5 $PIP_EXTRA_ARGS
}
else {
    Write-Host "Unsupported python version: $PY_VERSION, only 3.6 and 3.8 are supported"
    exit 1
}

Remove-Item -Path "$SDK_ROOT\$PY_ENV_NAME" -Recurse -Force -ErrorAction SilentlyContinue
Move-Item -Path (Join-Path $FILES_DIR $PY_ENV_NAME) -Destination (Join-Path $SDK_ROOT $PY_ENV_NAME) -Force

# Remove all unnecessary files
Remove-Item -Path $FILES_DIR -Recurse -Force -ErrorAction SilentlyContinue
